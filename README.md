# Railway Discord AI Voice Bot

A Python Discord voice bot using Discord voice receive + Gemini Live audio-to-audio.

## Important

This uses `discord-ext-voice-recv`, which is a community extension. Its upstream project warns that voice receive is not feature-complete and does not guarantee stability. The bot therefore includes reconnect/retry/error handling, but no software can guarantee zero errors.

## Commands

- `!join` — join your current VC and start listening
- `!leave` — leave VC
- `!stop` — stop current speech
- `!ping` — Discord gateway latency
- `!help_voice` — commands

## Environment variables

- `DISCORD_TOKEN` required
- `GEMINI_API_KEY` required
- `GEMINI_MODEL` optional; defaults to `gemini-3.8-live`
- `BOT_PERSONALITY` optional
- `LOG_LEVEL` optional

## Local run

1. Install Python 3.11.
2. Install FFmpeg and Opus on your machine.
3. `python -m venv .venv`
4. Activate it.
5. `pip install -r requirements.txt`
6. Copy `.env.example` to `.env` and fill in keys.
7. Export variables or load them with your preferred dotenv tool.
8. `python bot.py`

## Railway

Push the repository to GitHub. In Railway, create a project from the GitHub repository. Railway will detect the Dockerfile. Add `DISCORD_TOKEN` and `GEMINI_API_KEY` under the service Variables tab, then deploy.

Do not commit `.env` or API keys.
