# ESRGAN-Bot

A Discord bot for upscaling images with ESRGAN, plus a little more. NOTE: This bot only supports old architecture models and will NOT work with new architecture ones.

This version of the bot is a complete rewrite in python using Discord.py and the original source code by Xinntao, along with modifications from BlueAmulet's fork.

All credit to the original ESRGAN repository goes to Xinntao, and all credit to the original modifications goes to BlueAmulet.

Also many thanks to the GameUpscale Discord server for helping me test out this rewrite and for using the bot every day :)

You can see a working version of this bot in the GameUpscale discord server here: https://discord.gg/VR9SzTT

## Setup

### Configuration

A file in the root of this repository is needed called `config.yml`. This is required in order for the bot to work.

An example `config.json` looks like this:

```yml
bot_token: 'Discord Bot Token'
bot_prefix: '--'
img_size_cutoff: 1500
moderator_role_id: 549505502779015178
```

-   `bot_token` is the client token of your Discord bot.
-   `bot_prefix` is the prefix that will be used on all the bot's commands.
-   `img_size_cutoff` is the maximum resolution that you will allow someone to submit to the bot to avoid extended upscaling times.
-   `moderator_role_id` is the role id (can also be the name of the role) that will be allowed to run certain restricted commands.

### Models

Create a directory called `models` inside this directory, and place any models you want accessible to the bot in this folder.

## Running the bot

Run `testbot.py` like you would any other python script.
