const { DownloaderHelper } = require('node-downloader-helper');

module.exports = {
    downloadModel: function(url, destination, filename) {
        return new Promise((resolve, reject) => {
            const dl = new DownloaderHelper(url, destination, {
                fileName: filename,
                retry: { maxRetries: 2, delay: 10 },
                override: true
            });

            dl.on('end', () => {
                console.log('Download Completed');
                resolve(dl.getDownloadPath());
            }).on('error', () => {
                console.log('Download Failed');
                reject();
            });

            dl.start();
        });
    },
    downloadImage: function(url, destination) {
        return new Promise((resolve, reject) => {
            const dl = new DownloaderHelper(url, destination);

            dl.on('end', () => {
                console.log('Download Completed');
                resolve(dl.getDownloadPath());
            }).on('error', () => {
                console.log('Download Failed');
                reject();
            });

            dl.start();
        });
    }
};
