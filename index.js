const Discord = require("discord.js");
const client = new Discord.Client();

let request = require(`request`);
let fs = require(`fs`);
const fsExtra = require("fs-extra");

let {
    PythonShell
} = require("python-shell");

const queue = new Map();

function download(url, path) {
    request
        .get(url)
        .on("error", console.error)
        .pipe(fs.createWriteStream(path));
}

function checkURL(url) {
    let validUrl = require("valid-url");
    return (
        url.match(/\.(jpeg|jpg|png|JPEG|JPG|PNG)$/) != null &&
        validUrl.isUri(url)
    );
}

async function upscale(guild, item) {
    const serverQueue = queue.get(0); //queue.get(guild.id);

    if (!item) {
        queue.delete(0);
        //queue.delete(guild.id);
        return;
    }

    fsExtra.emptyDirSync("./ESRGAN/results/");
    fsExtra.emptyDirSync("./ESRGAN/LR/");

    let image = item.image;
    let model = item.model;
    let message = item.message;

    // downloads attachment image
    download(image.url, "./ESRGAN/LR/" + image.filename);

    // filename stuff
    let file = image.filename.split(".");
    let name = file[0];
    let ext = file[1];

    let modelName = model.includes(".pth") ? model : model + ".pth";

    let args = {
        args: [
            "./ESRGAN/models/" + modelName,
            "--input=./ESRGAN/LR/",
            "--output=./ESRGAN/results/"
        ]
    };
    PythonShell.run("./ESRGAN/test.py", args, function (err, results) {
        if (err) {
            console.log(err);
            queue.delete(0);
            return message.channel.send(
                "Sorry, there was an error processing your image."
            );
        }
        // results is an array consisting of messages collected during execution
        //console.log('results: %j', results);
        //client.channels
        //.get('66238856818589696')
        //.send(`${author},
        let filePath = `./ESRGAN/results/${name}_rlt.png`;

        try {
            if (!fs.existsSync(filePath)) {
                return message.channel.send(
                    "Sorry, there was an error processing your image."
                );
            }
        } catch (err) {
            return message.channel.send(
                "Sorry, there was an error processing your image."
            );
        }

        let stats = fs.statSync(filePath);
        let fileSizeInBytes = stats["size"];
        if (fileSizeInBytes >= 8000000) {
            let request = require("request");

            const options = {
                method: "POST",
                url: "http://0x0.st",
                //port: 443,
                headers: {
                    //Authorization: 'Basic ' + auth,
                    "Content-Type": "multipart/form-data"
                },
                formData: {
                    file: fs.createReadStream(
                        `./ESRGAN/results/${name}_rlt.png`
                    )
                }
            };

            request(options, function (err, res, body) {
                if (err) console.log(err);
                return message
                    .reply(`Resulting image too large, uploaded to: ${body}`)
                    .then(() => {
                        fsExtra.emptyDirSync("./ESRGAN/results/");
                        fsExtra.emptyDirSync("./ESRGAN/LR/");
                        serverQueue.jobs.shift();
                        try {
                            upscale(0, serverQueue.jobs[0]);
                        } catch (err) {
                            console.log(err);
                            queue.delete(0);
                            return message.channel.send(err);
                        }
                    })
                    .catch(err => {
                        console.log(err);
                    });
            });
        } else {
            return message
                .reply(`Upscaled using ${model}`, {
                    files: [`./ESRGAN/results/${name}_rlt.png`]
                })
                .then(() => {
                    fsExtra.emptyDirSync("./ESRGAN/results/");
                    fsExtra.emptyDirSync("./ESRGAN/LR/");
                    serverQueue.jobs.shift();
                    try {
                        upscale(0, serverQueue.jobs[0]);
                    } catch (err) {
                        console.log(err);
                        queue.delete(0);
                        return message.channel.send(err);
                    }
                })
                .catch(err => {
                    console.log(err);
                });
        }
    });
}

client.on("ready", () => {
    console.log(`Logged in as ${client.user.tag}!`);
    fsExtra.emptyDirSync("./ESRGAN/results/");
    fsExtra.emptyDirSync("./ESRGAN/LR/");
});

const prefix = "--";

client.on("message", async message => {
    if (!message.content.startsWith(prefix) || message.author.bot) return;

    const args = message.content.slice(prefix.length).split(" ");
    const command = args.shift().toLowerCase();

    const serverQueue = queue.get(0); //queue.get(message.guild.id);

    if (command === "upscale") {
        if (!args.length) {
            return message.channel.send(
                `You didn't provide any arguments, ${message.author}!`
            );
        } else {
            let image;
            let model;
            if (message.attachments.first()) {
                // limits filesize to 500kb
                if (message.attachments.first().filesize > 1500000) {
                    return message.reply(
                        "Sorry, that file is too large. Please send an image less than 1.5mb."
                    );
                }

                // limits dimensions to 500x500
                if (
                    message.attachments.first().width > 1000 ||
                    message.attachments.first().height > 1000
                ) {
                    return message.reply(
                        "Sorry, that file is too large. Please send an image less than 1000x1000."
                    );
                }

                // limits attachments to png or jpg/jpeg
                if (
                    !["png", "jpeg", "jpg"].includes(
                        message.attachments
                        .first()
                        .filename.split(".")[1]
                        .toLowerCase()
                    )
                ) {
                    return message.reply(
                        "Sorry, that file type is not supported."
                    );
                }

                image = message.attachments.first();
                model = args[0];
            } else if (checkURL(args[0])) {
                image = {
                    url: args[0],
                    filename: args[0].split("/").pop()
                };
                model = args[1];
            } else {
                return message.channel.send("Not a valid command.");
            }

            let modelName = model.includes(".pth") ? model : model + ".pth";
            if (
                !fs.readdirSync("./ESRGAN/models/").includes(modelName)
            ) {
                return message.channel.send("Not a valid model.");
            }

            let upscaleJob = {
                image: image,
                model: model,
                message: message
            };

            if (!serverQueue) {
                const queueContruct = {
                    textChannel: message.channel,
                    jobs: []
                };

                queue.set(0, queueContruct);
                //queue.set(message.guild.id, queueContruct);

                queueContruct.jobs.push(upscaleJob);

                try {
                    message.channel.send(`Your image is being processed.`);
                    upscale(message.guild, queueContruct.jobs[0]);
                } catch (err) {
                    console.log(err);
                    //queue.delete(message.guild.id);
                    queue.delete(0);
                    return message.channel.send(err);
                }
            } else {
                serverQueue.jobs.push(upscaleJob);
                //console.log(serverQueue.jobs);
                return message.channel.send(
                    `${image.filename} has been added to the queue! Your image is #${serverQueue.jobs.length} in line for processing.`
                );
            }
        }
    } else if (command === "models") {
        let files = fs.readdirSync("./ESRGAN/models/");
        let table = require("markdown-table");
        let models = [];
        for (let i = 0; i < files.length; i = i + 4) {
            models.push(files.slice(i, i + 4));
        }
        return message.channel.send(
            "```" + table(models, {
                rule: false
            }) + "```"
        );
        //return message.channel.send(files);
    } else if (command === "add") {
        let modelName = args[1].includes(".pth") ? args[1] : args[1] + ".pth";
        if (args[0].includes("drive.google.com")) {
            download(
                `https://docs.google.com/uc?export=download&id=${
                    args[0].split("/")[args[0].split("/").length - 2]
                }`,
                `./ESRGAN/models/${modelName}`
            );
        } else {
            download(args[0], `./ESRGAN/models/${modelName}`);
        }
    } else if (command === "help") {
        let help = `
Commands:

\`--upscale [model]\` // Upscales attached image using specified model

\`--upscale [url] [model]\` // Upscales linked image using specified model

\`--add [model url] [nickname]\` // Adds model from url, with a nickname (to avoid typing out long model names)

\`--models\` // Lists all models

\`--help\` // Shows this information again

Example: \`--upscale 4xBox.pth\``;
        return message.channel.send(help);
    }
});

client.on('error', console.error);

client.login("NjYzMjgwOTQ2ODYyMjI3NDU3.XhGOjA.pxpJl8c6ieUPkp3V-nOqKw21TzU");