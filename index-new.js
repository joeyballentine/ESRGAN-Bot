// Discord.js import
const Discord = require('discord.js');
const client = new Discord.Client();

const request = require(`request`);

// File stuff
const fs = require(`fs`);
const fsExtra = require('fs-extra');

// Image downloading stuff
const isImageUrl = require('is-image-url');
// const download = require('image-downloader');

// Imagemagick library
const gm = require('gm').subClass({
    imageMagick: true
});

// Python shell
const {
    PythonShell
} = require('python-shell');
const shell = require('shelljs');

// The image upscale queue
const queue = new Map();

// The prefix used for commands
const prefix = '!';

// Change these depending on what you want to allow
const pixelLimit = 500 * 500;
const sizeLimit = 500000;

// Path to ESRGAN. Should be initialized by a submodule and is meant to be used with BlueAmulet's fork
const esrganPath = './ESRGAN/';

// Connects to the bot account and empties the directories
client.on('ready', () => {
    console.log(`Logged in as ${client.user.tag}!`);
    emptyDirs();
});

// Message event handler
client.on('message', async message => {
    // Removes extra spaces between commands
    message.content = message.content.replace(/ +(?= )/g, '');

    // The bot will not respond unless the prefix is typed
    // It will also ignore anything sent by itself
    if (!message.content.startsWith(prefix) || message.author.bot) return;

    // Splits the args into an array
    const args = message.content.slice(prefix.length).split(' ');

    // Strips the command off of the args array
    const command = args.shift().toLowerCase();

    // Does all the steps required for upscaling the image
    if (command === 'upscale') {
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
            return message.channel.send('Not a valid command.');
        }

        // Sanitizing the url
        url = url.split('&')[0];
        url = url.split('?')[0];

        // Downloads the image
        let image = url.split("/").pop()
        download(url, image);

        // Gets the model name from the model argument
        let model = args[0].includes('.pth') ? args[0] : args[0] + '.pth';

        // Checks to make sure model name is valid (exists and is spelled right)
        if (!fs.readdirSync(esrganPath + '/models/').includes(model)) {
            return message.channel.send('Not a valid model.');
        }

        // The job sent for processing
        let upscaleJob = {
            model: model,
            image: image,
            resize: false,
            filter: false,
            montage: false,
            message: message
        };

        // Parsing the extra arguments

        // Resize
        //upscaleJob.resize = args[args.indexOf(arg) + 1];
        // upscaleJob.resize = ['--resize', '-r'].some(arg => {
        //     console.log('resize!')
        //     if (args.includes(arg)) {
        //         console.log(args[args.indexOf(arg) + 1])
        //         return args[args.indexOf(arg) + 1]
        //     };
        // })

        if (args.includes('--resize')) {
            upscaleJob.resize = args[args.indexOf('--resize') + 1]
        }
        console.log(upscaleJob.resize);

        // filter
        if (resize && ['--filter', '-f'].some(arg => args.includes(arg))) {
            upscaleJob.filter = args[args.indexOf(arg) + 1];
        }

        // Montage
        if (['--montage', '-m'].some(arg => args.includes(arg))) {
            upscaleJob.montage = args[args.indexOf(arg) + 1];
        }

        // Checks if the image is valid to be upscaled
        if (checkImage(image)) {
            // Adds to the queue and starts upscaling if not already started.
            if (!queue.get(0)) {
                const queueConstruct = {
                    textChannel: message.channel,
                    jobs: []
                };
                queue.set(0, queueConstruct);
                queue.get(0).jobs.push(upscaleJob);

                try {
                    message.channel.send(`Your image is being processed.`);
                    process(queue.get(0).jobs[0]);
                } catch (err) {
                    // If something goes wrong here we just reset the entire queue
                    // This probably isn't ideal but it's what the music bots do
                    console.log(err);
                    queue.delete(0);
                    return message.channel.send(err);
                }
            } else {
                queue.get(0).jobs.push(upscaleJob);
                return message.channel.send(
                    `${image} has been added to the queue! Your image is #${queue.length} in line for processing.`
                );
            }
        } else {
            return message.channel.send(
                `Sorry, that image cannot be processed.`
            );
        }
    }
});

client.login('NjYzMTA3NTQ3OTg4NzU0NDUy.XhDtGg.CGxZaTJRr7OmYJOVbBlY2j9bspc');

function emptyDirs() {
    fsExtra.emptyDirSync(esrganPath + '/results/');
    fsExtra.emptyDirSync(esrganPath + '/LR/');
}

// async function downloadImage(url) {
//     const options = {
//         url: url,
//         dest: esrganPath + './LR'
//     };

//     let image = await download
//         .image(options)
//         .then(({ filename, image }) => {
//             return filename.split(`\\`).pop();
//         })
//         .catch(err => console.error(err));

//     return image;
// }

function download(url, filename) {
    request
        .get(url)
        .on('error', console.error)
        .pipe(fs.createWriteStream(esrganPath + '/LR/' + filename));
}

function checkImage(image) {
    if (
        ['png', 'jpg', 'jpeg'].some(
            filetype =>
            image.split('.').pop() === filetype.toLowerCase()
        )
    ) {
        return true;
    } else return false;
}

function process(job) {
    if (job.resize) resize(job.image, job.resize, job.filter);

    //split();
    upscale(job.image, job.model, job, (job) => {
        //merge();
        optimize();

        if (job.montage) montage(job.image, job.model, job.message);

        return job.message
            .reply(`Upscaled using ${job.model}`, {
                files: [`${esrganPath}/results/${job.image.split('.')[0]}_rlt.png`]
            })
            .then(() => {
                queue.get(0).jobs.shift();
                try {
                    if (queue.get(0).jobs.length > 0) {
                        process(queue.get(0).jobs[0]);
                    } else {
                        queue.delete(0);
                    }
                } catch (err) {
                    console.log(err);
                    queue.delete(0);
                    return job.message.channel.send(err);
                }
            });
    });

}

function upscale(image, model, job, callback) {
    let args = {
        args: [
            `${esrganPath}/models/${model}`,
            `--input=${esrganPath}/LR/`,
            `--output=${esrganPath}/results/`
        ]
    };
    PythonShell.run(esrganPath + '/test.py', args, (err, results) => {
        if (err) {
            console.log(err);
            queue = [];
            return message.channel.send(
                'Sorry, there was an error processing your image.'
            );
        }

        let filePath = `${esrganPath}/results/${image.split('.')[0]}_rlt.png`;

        try {
            if (!fs.existsSync(filePath)) {
                return message.channel.send(
                    'Sorry, there was an error processing your image.'
                );
            } else {
                callback(job);
            }
        } catch (err) {
            return message.channel.send(
                'Sorry, there was an error processing your image.'
            );
        }
    });
}

function resize(image, amount, filter) {
    gm(`${esrganPath}/LR/${image}`)
        .resize((1.0 / amount) * 100.0 + '%')
        .filter(filter)
        .write(`${esrganPath}/LR/${image}`, function (err) {
            if (err) console.log(err);
            if (!err) console.log('done');
        });
}

function montage(image, model, message) {
    //TODO extract image % difference for scaling
    shell.exec(
        `./scripts/montage.sh -if="${esrganPath}/LR/${image.filename}" -is="${esrganPath}/results/${image.filename}_rlt" -tf="LR" -ts="${model}" -td="2x1" -uf="400%" -ug="100%" -io="output_montage.png"`,
        () => {
            return message.channel.send('', {
                files: [`output_montage.png.png`]
            });
        }
    );
}

function split() {
    shell.exec(`./scripts/split.sh`);
}

function merge() {
    shell.exec(`./scripts/merge.sh`);
}

function optimize() {
    const imagemin = require('imagemin');
    const imageminOptipng = require('imagemin-optipng');

    (async () => {
        await imagemin(
            [`${esrganPath}/results/*.png`],
            `${esrganPath}/results/`, {
                use: [imageminOptipng()]
            }
        );

        console.log('Images optimized!');
    })();
}