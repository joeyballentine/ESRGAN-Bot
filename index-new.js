// Discord.js import
const Discord = require("discord.js");
const client = new Discord.Client();

const request = require(`request`);

// File stuff
const fs = require(`fs`);
const fsExtra = require("fs-extra");

// Image downloading stuff
const isImageUrl = require("is-image-url");
const download = require("image-downloader");

// Python shell
const { PythonShell } = require("python-shell");

// The image upscale queue
// This uses an array instead of a map as it must be the same queue across all servers
const queue = new Array();

// The prefix used for commands
const prefix = "--";

// Change these depending on what you want to allow
const pixelLimit = 500 * 500;
const sizeLimit = 500000;

// Path to ESRGAN. Should be initialized by a submodule and is meant to be used with BlueAmulet's fork
const esrganPath = "./ESRGAN/";

// Connects to the bot account and empties the directories
client.on("ready", () => {
    console.log(`Logged in as ${client.user.tag}!`);
    emptyDirs();
});

// Message event handler
client.on("message", async message => {
    // Removes extra spaces between commands
    message = message.replace(/ +(?= )/g, "");

    // The bot will not respond unless the prefix is typed
    // It will also ignore anything sent by itself
    if (!message.content.startsWith(prefix) || message.author.bot) return;

    // Splits the args into an array
    const args = message.content.slice(prefix.length).split(" ");

    // Strips the command off of the args array
    const command = args.shift().toLowerCase();

    // Does all the steps required for upscaling the image
    if (command === "upscale") {
        // If no args are given the bot will stop and send an error message
        if (!args.length) {
            return message.channel.send(
                `You didn't provide any arguments, ${message.author}!`
            );
        }

        // Grabs the url of the image whether its an attachment or a url
        let url;
        if (message.attachments.first()) {
            url = message.attachments.first().url;
        } else if (isImageUrl(args[0])) {
            // Strips the url off of the args if a url is given
            url = args.shift();
        } else {
            // If no image is given the bot will error
            return message.channel.send("Not a valid command.");
        }

        // Downloads the image and returns an filename & image
        let image = downloadImage(url);

        // Gets the model name from the model argument
        let model = args[0].includes(".pth") ? args[0] : args[0] + ".pth";

        // Checks to make sure model name is valid (exists and is spelled right)
        if (!fs.readdirSync(esrganPath + "/models/").includes(model)) {
            return message.channel.send("Not a valid model.");
        }

        // The job sent for processing
        let upscaleJob = {
            model: model,
            image: image,
            resize: false,
            filter: false,
            montage: false
        };

        // Parsing the extra arguments

        // Resize
        if (["--resize", "-r"].some(arg => args.includes(arg))) {
            upscaleJob.resize = args[args.indexOf(arg) + 1];
        }

        // filter
        if (resize && ["--filter", "-f"].some(arg => args.includes(arg))) {
            upscaleJob.filter = args[args.indexOf(arg) + 1];
        }

        // Montage
        if (["--montage", "-m"].some(arg => args.includes(arg))) {
            upscaleJob.montage = args[args.indexOf(arg) + 1];
        }

        // Checks if the image is valid to be upscaled
        if (checkImage(image)) {
            // Adds to the queue and starts upscaling if not already started.
            if (!queue) {
                queue.push(upscaleJob);

                try {
                    message.channel.send(`Your image is being processed.`);
                    process(queue[0]);
                } catch (err) {
                    // If something goes wrong here we just reset the entire queue
                    // This probably isn't ideal but it's what the music bots do
                    console.log(err);
                    queue = [];
                    return message.channel.send(err);
                }
            } else {
                queue.push(upscaleJob);
                return message.channel.send(
                    `${image.filename} has been added to the queue! Your image is #${queue.length} in line for processing.`
                );
            }
        } else {
            return message.channel.send(
                `Sorry, that image cannot be processed.`
            );
        }
    }
});

function emptyDirs() {
    fsExtra.emptyDirSync(esrganPath + "/results/");
    fsExtra.emptyDirSync(esrganPath + "/LR/");
}

function downloadImage(url) {
    const options = {
        url: url,
        dest: esrganPath + "./LR"
    };

    download
        .image(options)
        .then(({ filename, image }) => {
            console.log("Saved to", filename);
            console.log(image);
            return {
                filename,
                image
            };
        })
        .catch(err => console.error(err));
}

function process(job) {
    if (job.resize) resize(job.resize, job.filter);

    //split();
    upscale(job.model);
    //merge();
    optimize();

    if (job.montage) montage();
    return;
}

function upscale() {
    // Clear folders (just in case)
    emptyDirs();
}

function resize() {}

function montage() {}

function split() {
    //TODO
}

function merge() {
    //TODO
}

function optimize() {}
