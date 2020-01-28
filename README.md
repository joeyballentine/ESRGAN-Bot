# ESRGAN-Bot

A Discord bot for upscaling images with ESRGAN, plus a little more.

## Setup

Clone this repository and initialize the submodule. The included submodule is BlueAmulet's ESRGAN fork, which is technically required for the bot to work. Theoretically you could use another version of ESRGAN, but you would then only be able to use 4x models without further code modification.

### Node

If you already have node.js installed, great! If not, get it here: https://nodejs.org/en/

Once you have the repository cloned, simply type `npm i` or `npm install` in a terminal inside the directory to install all the node dependencies.

### Configuration

A file in the root of this repository is needed called `config.json`. This is required in order for the bot to work.

An example `config.json` looks like this:

```
{
    "token": "Discord Bot Token",
    "prefix": "!",
    "pixelLimit": 1000,
    "esrganPath": "./ESRGAN/"
}
```

-   `token` is the client token of your Discord bot.
-   `prefix` is the prefix that will be used on all the bot's commands.
-   `pixelLimit` is the number of pixels (both width and height) that will trigger the splitting/merging functionality. This is widely system dependent as the amount of VRAM your GPU has determines how large of an image ESRGAN can process at once. you may need to play with this number a bit to see what works for you.
-   `esrganPath` is the path where ESRGAN is located. If you initialized the submodule, `'./ESRGAN/'` will work fine.

### Extra Dependencies

Alongside the npm modules and ESRGAN itself, you will need bash and ImageMagick for all functionality to work.

If you are using Windows, please install Git for Windows to gain bash functionality.You can get that here: https://gitforwindows.org/

ImageMagick is an image processing tool. You can download ImageMagick here: https://imagemagick.org/script/download.php

## Running the bot

Type `node index.js` to start the bot. If everything was succcessful, you should see `Logged in as [Bot Name]!` printed in the console.
