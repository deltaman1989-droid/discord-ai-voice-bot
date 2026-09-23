import asyncio
import logging
import os
import signal
from typing import Optional

import discord
from discord.ext import commands, voice_recv
from google import genai
from google.genai import types

from audio import pcm48_stereo_to_16k_mono, GeminiAudioSource

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("voicebot")

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-live")

if not DISCORD_TOKEN:
    raise RuntimeError("DISCORD_TOKEN is missing")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is missing")

# Discord.py's voice stack needs Opus available on Linux.
discord.opus._load_default()

intents = discord.Intents.default()
intents.message_content = True
intents.voice_states = True

bot = commands.Bot(command_prefix="!", intents=intents, help_command=None)
gemini = genai.Client(api_key=GEMINI_API_KEY)

SYSTEM_PROMPT = os.getenv(
    "BOT_PERSONALITY",
    """You are a friendly Discord voice companion. Speak naturally and conversationally.
Keep replies concise because you are talking in a voice channel. Usually answer in 1-3 short sentences.
Do not mention being an AI unless directly asked. Do not narrate your internal reasoning.
If several people are talking, respond to the most recent clear speaker. Be friendly, casual, and helpful.""",
)


class GuildVoiceSession:
    def __init__(self, guild: discord.Guild, voice_client: voice_recv.VoiceRecvClient):
        self.guild = guild
        self.vc = voice_client
        self.loop = asyncio.get_running_loop()
        self.audio_queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=100)
        self.playback_queue: asyncio.Queue[bytes | None] = asyncio.Queue(maxsize=300)
        self.session = None
        self.sender_task: Optional[asyncio.Task] = None
        self.receiver_task: Optional[asyncio.Task] = None
        self.play_task: Optional[asyncio.Task] = None
        self.source: Optional[GeminiAudioSource] = None
        self.closed = False
        self._send_lock = asyncio.Lock()

    async def start(self):
        config = types.LiveConnectConfig(
            response_modalities=[types.Modality.AUDIO],
            system_instruction=types.Content(
                role="user", parts=[types.Part(text=SYSTEM_PROMPT)]
            ),
            # Let Gemini's server-side VAD decide when a user has finished speaking.
            realtime_input_config=types.RealtimeInputConfig(
                automatic_activity_detection=types.AutomaticActivityDetection(
                    silence_duration_ms=650,
                )
            ),
            input_audio_transcription=types.AudioTranscriptionConfig(),
            output_audio_transcription=types.AudioTranscriptionConfig(),
        )

        self.session = await self._connect(config)
        self.sender_task = asyncio.create_task(self._send_loop(), name=f"send-{self.guild.id}")
        self.receiver_task = asyncio.create_task(self._receive_loop(), name=f"recv-{self.guild.id}")
        self.play_task = asyncio.create_task(self._play_loop(), name=f"play-{self.guild.id}")

        self.source = GeminiAudioSource(self.playback_queue)
        if not self.vc.is_playing():
            self.vc.play(self.source, after=self._playback_done)

    async def _connect(self, config):
        last_error = None
        for attempt in range(1, 6):
            try:
                log.info("Connecting Gemini Live for guild %s (attempt %s)", self.guild.id, attempt)
                cm = gemini.aio.live.connect(model=GEMINI_MODEL, config=config)
                session = await cm.__aenter__()
                self._session_cm = cm
                log.info("Gemini Live connected for guild %s", self.guild.id)
                return session
            except Exception as exc:
                last_error = exc
                log.exception("Gemini connection failed")
                await asyncio.sleep(min(2 ** attempt, 10))
        raise RuntimeError(f"Could not connect to Gemini Live: {last_error}")

    def feed_from_thread(self, pcm48_stereo: bytes):
        if self.closed:
            return
        pcm16k = pcm48_stereo_to_16k_mono(pcm48_stereo)
        if not pcm16k:
            return

        def put():
            if self.closed:
                return
            try:
                self.audio_queue.put_nowait(pcm16k)
            except asyncio.QueueFull:
                # Drop the oldest audio rather than letting the bot's latency grow forever.
                try:
                    self.audio_queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
                try:
                    self.audio_queue.put_nowait(pcm16k)
                except asyncio.QueueFull:
                    pass

        self.loop.call_soon_threadsafe(put)

    async def _send_loop(self):
        while not self.closed:
            pcm = await self.audio_queue.get()
            if self.closed or not self.session:
                continue
            try:
                await self.session.send_realtime_input(
                    audio=types.Blob(data=pcm, mime_type="audio/pcm;rate=16000")
                )
            except Exception:
                log.exception("Gemini audio send failed for guild %s", self.guild.id)
                # Receiver loop will generally notice a dead session. Do not crash the bot.

    async def _receive_loop(self):
        try:
            async for response in self.session.receive():
                if self.closed:
                    break

                if response.server_content:
                    sc = response.server_content
                    if sc.input_transcription and sc.input_transcription.text:
                        log.info("USER[%s]: %s", self.guild.id, sc.input_transcription.text.strip())
                    if sc.output_transcription and sc.output_transcription.text:
                        log.info("BOT[%s]: %s", self.guild.id, sc.output_transcription.text.strip())

                # Live API audio is normally 24 kHz PCM. GeminiAudioSource converts it to
                # the 48 kHz stereo PCM format Discord expects.
                data = getattr(response, "data", None)
                if data:
                    try:
                        self.playback_queue.put_nowait(data)
                    except asyncio.QueueFull:
                        # Clear queued speech to preserve low latency, then add the latest chunk.
                        while not self.playback_queue.empty():
                            try:
                                self.playback_queue.get_nowait()
                            except asyncio.QueueEmpty:
                                break
                        try:
                            self.playback_queue.put_nowait(data)
                        except asyncio.QueueFull:
                            pass
        except asyncio.CancelledError:
            raise
        except Exception:
            if not self.closed:
                log.exception("Gemini receive loop died for guild %s", self.guild.id)
                await self._reconnect()

    async def _reconnect(self):
        if self.closed:
            return
        # A reconnect is deliberately serialized so several simultaneous failures don't
        # create multiple Live sessions.
        async with self._send_lock:
            if self.closed:
                return
            try:
                old_cm = getattr(self, "_session_cm", None)
                if old_cm:
                    try:
                        await old_cm.__aexit__(None, None, None)
                    except Exception:
                        pass

                config = types.LiveConnectConfig(
                    response_modalities=[types.Modality.AUDIO],
                    system_instruction=types.Content(
                        role="user", parts=[types.Part(text=SYSTEM_PROMPT)]
                    ),
                    realtime_input_config=types.RealtimeInputConfig(
                        automatic_activity_detection=types.AutomaticActivityDetection(
                            silence_duration_ms=650,
                        )
                    ),
                    input_audio_transcription=types.AudioTranscriptionConfig(),
                    output_audio_transcription=types.AudioTranscriptionConfig(),
                )
                self.session = await self._connect(config)
                log.info("Gemini Live reconnected for guild %s", self.guild.id)
                if self.receiver_task and not self.receiver_task.done():
                    self.receiver_task.cancel()
                self.receiver_task = asyncio.create_task(self._receive_loop())
            except Exception:
                log.exception("Gemini reconnect failed")

    async def _play_loop(self):
        # The AudioSource itself blocks in Discord's audio thread, so this task's job is simply
        # to feed resampled PCM into its thread-safe queue.
        while not self.closed:
            data = await self.playback_queue.get()
            if data is None:
                break
            if self.source:
                self.source.push_gemini_pcm(data)

    def _playback_done(self, error):
        if error:
            log.error("Discord playback error in guild %s: %r", self.guild.id, error)

    async def stop(self):
        if self.closed:
            return
        self.closed = True
        for task in (self.sender_task, self.receiver_task, self.play_task):
            if task:
                task.cancel()
        try:
            if self.vc.is_playing():
                self.vc.stop()
        except Exception:
            pass
        try:
            await self.vc.disconnect(force=True)
        except Exception:
            pass
        cm = getattr(self, "_session_cm", None)
        if cm:
            try:
                await cm.__aexit__(None, None, None)
            except Exception:
                pass


class ReceiveSink(voice_recv.AudioSink):
    def __init__(self, session: GuildVoiceSession):
        super().__init__()
        self.session = session

    def wants_opus(self) -> bool:
        return False

    def write(self, user, data):
        if user is None or user == bot.user:
            return
        try:
            self.session.feed_from_thread(data.pcm)
        except Exception:
            log.exception("Voice receive callback failed")

    def cleanup(self):
        pass


sessions: dict[int, GuildVoiceSession] = {}


@bot.event
async def on_ready():
    log.info("Logged in as %s (%s)", bot.user, bot.user.id)
    log.info("Invite the bot with Connect, Speak, and Use Voice Activity permissions.")


@bot.event
async def on_voice_state_update(member, before, after):
    # If Discord disconnects the bot from a voice channel, clean up its session.
    if member.id != bot.user.id:
        return
    if before.channel and not after.channel:
        session = sessions.pop(before.channel.guild.id, None)
        if session:
            await session.stop()


@bot.command()
@commands.guild_only()
async def join(ctx: commands.Context):
    """Join your current VC and start listening."""
    if not ctx.author.voice or not ctx.author.voice.channel:
        await ctx.send("Join a voice channel first.")
        return

    channel = ctx.author.voice.channel
    existing = sessions.get(ctx.guild.id)
    if existing and existing.vc.channel == channel:
        await ctx.send("I'm already in this voice channel.")
        return

    if existing:
        await existing.stop()
        sessions.pop(ctx.guild.id, None)

    try:
        vc = await channel.connect(cls=voice_recv.VoiceRecvClient, reconnect=True)
        session = GuildVoiceSession(ctx.guild, vc)
        sessions[ctx.guild.id] = session
        vc.listen(ReceiveSink(session))
        await session.start()
        await ctx.send(f"Joined **{channel.name}**. Start talking.")
    except Exception as exc:
        sessions.pop(ctx.guild.id, None)
        log.exception("Join failed")
        try:
            if ctx.voice_client:
                await ctx.voice_client.disconnect(force=True)
        except Exception:
            pass
        await ctx.send(f"I couldn't start voice mode: `{type(exc).__name__}`")


@bot.command()
@commands.guild_only()
async def leave(ctx: commands.Context):
    """Leave the current VC."""
    session = sessions.pop(ctx.guild.id, None)
    if session:
        await session.stop()
    elif ctx.voice_client:
        await ctx.voice_client.disconnect(force=True)
    await ctx.send("Left the voice channel.")


@bot.command()
@commands.guild_only()
async def stop(ctx: commands.Context):
    """Stop speaking; keep listening."""
    session = sessions.get(ctx.guild.id)
    if session and session.vc.is_playing():
        session.vc.stop()
        session.source = GeminiAudioSource(session.playback_queue)
        session.vc.play(session.source, after=session._playback_done)
    await ctx.send("Stopped my current speech.")


@bot.command()
@commands.guild_only()
async def ping(ctx: commands.Context):
    await ctx.send(f"Pong — `{round(bot.latency * 1000)} ms` gateway latency.")


@bot.command()
@commands.guild_only()
async def help_voice(ctx: commands.Context):
    await ctx.send(
        "**Voice commands:** `!join` `!leave` `!stop` `!ping`\n"
        "Join a VC, then just talk normally."
    )


async def shutdown():
    log.info("Shutting down...")
    for session in list(sessions.values()):
        await session.stop()
    sessions.clear()
    if not bot.is_closed():
        await bot.close()


def install_signal_handlers():
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, lambda: asyncio.create_task(shutdown()))
        except (NotImplementedError, RuntimeError):
            pass


async def main():
    install_signal_handlers()
    await bot.start(DISCORD_TOKEN)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
