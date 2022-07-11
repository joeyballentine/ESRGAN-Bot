# ESRGAN-Bot

NOTE: This hasn't been super actively maintained for a while. If you plan on running this yourself keep in mind that I have a few things hardcoded and once discord enforces slash commands, this bot will no longer work.

A Discord bot for upscaling images with ESRGAN (and variations including ESRGAN+, RealSR, BSRGAN, Real-ESRGAN), plus a little more.

This version of the bot is a complete rewrite in python using Discord.py and the original source code by Xinntao, along with modifications from BlueAmulet's fork.

All credit to the original ESRGAN repository goes to Xinntao, and all credit to the original modifications goes to BlueAmulet.

Also many thanks to the GameUpscale Discord server for helping me test out this rewrite and for using the bot every day :)

You can see a working version of this bot in the GameUpscale discord server here: https://discord.gg/VR9SzTT

## Setup

### Installation

Setup a python virtual environment: `py -m venv venv`

Enable the virtual environment:
- Windows: `venv/scripts/activate`
- Linux: `source venv/bin/activate`

Install requirements: `pip install -r requirements.txt`

Visit the PyTorch website here and generate the correct command for your system.
For Package hit "Pip", for Language hit "Python".
If you are using a modern Nvidia card with up-to-date drivers select the latest CUDA for Compute Platform.
If using AMD or no GPU at all, hit CPU instead.
Copy the command generated, run it in terminal, installation complete!

### Configuration

Rename or copy the "config.yml.dist" file to "config.yml".

Edit this file appropriately:
-   `bot_token` is the client token of your Discord bot.
-   `bot_prefix` is the prefix that will be used on all the bot's commands.
-   `img_size_cutoff` is the maximum resolution that you will allow someone to submit to the bot to avoid extended upscaling times.
-   `moderator_role_id` is the role id (can also be the name of the role) that will be allowed to run certain restricted commands.
-   `global_guild_block`, keep this false if you want the bot to be able to run in any server.

### Models

Create a directory called `models` inside this directory, and place any models you want accessible to the bot in this folder.

## Running the bot

Run `testbot.py` like you would any other python script.
