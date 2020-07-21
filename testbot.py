import argparse
import glob
import math
import os.path
import re
import sys
import urllib.request
from io import BytesIO
from os import walk

import cv2
import discord
import fuzzywuzzy
import numpy as np
import torch
import yaml
from discord.ext import commands
from fuzzywuzzy import fuzz, process

import architecture as arch

description = '''A rewrite of the ESRGAN bot entirely in python'''

try:
    config = yaml.safe_load(open('./config.yml'))
except:
    print('You must provide a config.yml!!!')

bot = commands.Bot(command_prefix=config['bot_prefix'],
                   description=description)
bot.remove_command('help')


@bot.check
async def globally_block_not_gu(ctx):
    is_dm = ctx.guild is None
    if is_dm:
        print(f'DM, {ctx.author.name}')
        await ctx.message.channel.send(
            '{}, ESRGAN bot is not permitted for use in DMs. Please join the GameUpscale server at discord.gg/VR9SzTT to continue use of this bot. Thank you.'.format(ctx.author.mention))
        return False
    else:
        is_gu = ctx.guild.id == 547949405949657098
        if not is_gu:
            print(f'{ctx.guild.name}, {ctx.author.name}')
            await ctx.message.channel.send(
                '{}, ESRGAN bot is not permitted for use in this server. Please join the GameUpscale server at discord.gg/VR9SzTT to continue use of this bot. Thank you.'.format(ctx.author.mention))
            return False
        return True


# @bot.event
# async def on_command_error(ctx, error):
#     if isinstance(error, commands.CheckFailure):
#         return
#     else:
#         print(error)


class ESRGAN(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        self.queue = {}

        # This group of variables are used in the upscaling process
        self.last_model = None
        self.last_in_nc = None
        self.last_out_nc = None
        self.last_nf = None
        self.last_nb = None
        self.last_scale = None
        self.model = None
        self.device = torch.device('cuda')

        # This group of variables pertain to the models list
        self.models = []
        for (dirpath, dirnames, filenames) in walk('./models'):
            self.models.extend(filenames)
            break
        self.fuzzymodels, self.aliases = self.build_aliases()

    @commands.Cog.listener()
    async def on_ready(self):
        print(f'Logged in as {self.bot.user.name} - {self.bot.user.id}')
        print('------')
        await bot.change_presence(status=discord.Status.online, activity=discord.Game("ESRGAN"))

    @commands.command()
    async def help(self, ctx):
        await ctx.message.channel.send('''Commands:

`{0}upscale [model]` // Upscales attached image using specified model. Model name input will be automatically matched with the closest model name.

`{0}upscale [url] [model]` // Upscales linked image using specified model. Model name input will be automatically matched with the closest model name.

`{0}models [-dm]` // Lists all available models as a .txt file. Can add `-dm` flag to be DM'd a regular list.

`{0}addmodel [url] [nickname]` // Adds model from url with nickname. Please include the scale of the model in the name, for example, 4xBox.

`{0}montage [url1] [url2] [label]` // Creates a montage of two specified images with specified label

`{0}downscale [url] [amount] [filter]` // Downscales the image by the amount listed. Filter is optional and defaults to box/area.

`{0}help` // Lists this information again.

Optional `{0}upscale` args:

`-downscale [amount]` // Downscales the image by the amount listed. For example, `-downscale 4` will make the image 25% of its original size.

`-filter [filter]` // Filter to be used for downscaling. Must be a valid OpenCV image interpolation filter, with ImageMagick aliases supported as well. Defaults to box/area.

`-blur [type] [amount]` // Blurs the image before upscaling using the specified blur type and the amount specified. Only Gaussian and median blur are currently supported.

`-montage` // Creates aside by side comparison of the LR and result after upscaling.

`-seamless` // Duplicates the image around the edges to make a seamless texture retain its seamlessness

Example: `{0}upscale www.imageurl.com/image.png 4xBox.pth -downscale 4 -filter point -montage`'''.format(bot.command_prefix))

    @commands.command()
    async def models(self, ctx, *args):
        if '-dm' in args:
            models = iter([m[:-4] for m in self.models])
            plus = False
            sublists, current = [], f'+ {next(models)}'
            for model in models:
                if len(current) + 4 + len(model) > 1990:
                    sublists.append(current)
                    current = model
                else:
                    current += '\n{} {}'.format(('+' if plus else '-'),
                                                model)
                    plus = not plus
            sublists.append(current)
            for sublist in sublists:
                message = f'```diff\n{sublist}```'
                await ctx.author.send(message)

        else:
            models = [m[:-4] for m in self.models]
            await ctx.message.channel.send('', file=discord.File(BytesIO(('\n'.join(models)).encode('utf8')), 'models.txt'))

    @commands.command()
    async def addmodel(self, ctx, url, nickname):
        await ctx.message.channel.send('Adding model {}...'.format(nickname))
        model_name = nickname.replace('.pth', '') + '.pth'
        try:
            if not os.path.exists('./models/{}'.format(model_name)):
                urllib.request.urlretrieve(
                    url, './models/{}'.format(model_name))
                await ctx.message.channel.send('Model {} successfully added.'.format(nickname))
                self.models.append(model_name)
                self.models.sort()
                self.fuzzymodels, self.aliases = self.build_aliases()
            else:
                await ctx.message.channel.send('Model {} already exists.'.format(nickname))
        except:
            await ctx.message.channel.send('Error adding model {}!'.format(nickname))

    @commands.command()
    @commands.has_role(config['moderator_role_id'])
    async def replacemodel(self, ctx, url, nickname):
        await ctx.message.channel.send('Replacing model {}...'.format(nickname))
        model_name = nickname.replace('.pth', '') + '.pth'
        try:
            if os.path.exists('./models/{}'.format(model_name)):
                urllib.request.urlretrieve(
                    url, './models/{}'.format(model_name))
                await ctx.message.channel.send('Model {} successfully added.'.format(nickname))
                self.models.append(model_name)
                self.models.sort()
                self.fuzzymodels, self.aliases = self.build_aliases()
            else:
                await ctx.message.channel.send('Model {} does not exist.'.format(nickname))
        except:
            await ctx.message.channel.send('Error replacing model {}!'.format(nickname))

    @commands.command()
    @commands.has_role(config['moderator_role_id'])
    async def removemodel(self, ctx, nickname):
        model_name = nickname.replace('.pth', '') + '.pth'
        if model_name in self.models and os.path.exists('./models/{}'.format(model_name)):
            self.models.remove(model_name)
            self.models.sort()
            os.unlink('./models/{}'.format(model_name))
            self.fuzzymodels, self.aliases = self.build_aliases()
            await ctx.message.channel.send('Removed model {}!'.format(nickname))
        else:
            await ctx.message.channel.send('Model {} doesn\'t exist!'.format(nickname))

    @commands.command()
    @commands.has_role(config['moderator_role_id'])
    async def reloadmodels(self, ctx):
        self.models = []
        for (dirpath, dirnames, filenames) in walk('./models'):
            self.models.extend(filenames)
        self.models.sort()
        self.fuzzymodels, self.aliases = self.build_aliases()
        await ctx.message.channel.send('Done.')

    @removemodel.error
    @replacemodel.error
    async def not_mod_handler(self, ctx, error):
        if isinstance(error, commands.MissingRole):
            await ctx.message.channel.send('You do not have permission to perform that command!')

    @commands.command()
    async def montage(self, ctx, img1, img2, label):
        try:
            image_1 = self.download_img_from_url(img1)
            image_2 = self.download_img_from_url(img2)
        except:
            await ctx.message.channel.send('{}, one of your images could not be downloaded.'.format(ctx.message.author.mention))
            return

        await ctx.message.channel.send('Creating montage...'.format(ctx.message.author.mention))

        try:
            montage = self.make_montage(image_1, image_2, label)
        except:
            await ctx.message.channel.send('{}, there was an error creating your montage.'.format(ctx.message.author.mention))
            return

        data = BytesIO(cv2.imencode('.png', montage, [
            cv2.IMWRITE_PNG_COMPRESSION, 16])[1].tostring())

        await ctx.message.channel.send('', file=discord.File(data, 'montage.png'))

    @commands.command()
    async def downscale(self, ctx, img, amt, filter='area'):
        try:
            image = self.download_img_from_url(img)
            filename = img[img.rfind("/")+1:]
        except:
            await ctx.message.channel.send('{}, your images could not be downloaded.'.format(ctx.message.author.mention))
            return

        await ctx.message.channel.send('Downscaling image...')

        if filter in {'point', 'nearest', 'nn', 'nearest_neighbor', 'nearestneighbor'}:
            interpolation = cv2.INTER_NEAREST
        elif filter in {'box', 'area'}:
            interpolation = cv2.INTER_AREA
        elif filter in {'linear', 'bilinear', 'triangle'}:
            interpolation = cv2.INTER_LINEAR
        elif filter in {'cubic', 'bicubic'}:
            interpolation = cv2.INTER_CUBIC
        elif filter in {'exact', 'linear_exact', 'linearexact'}:
            interpolation = cv2.INTER_LINEAR_EXACT
        elif filter in {'lanczos', 'lanczos64'}:
            interpolation = cv2.INTER_LANCZOS4
        else:
            interpolation = cv2.INTER_AREA

        image = self.downscale_img(image, interpolation, amt)

        data = BytesIO(cv2.imencode('.png', image, [
            cv2.IMWRITE_PNG_COMPRESSION, 16])[1].tostring())

        await ctx.message.channel.send('{}, your image has been downscaled by {}.'.format(ctx.author.mention, amt), file=discord.File(data, '{}.png'.format(filename.split('.')[0])))

    @commands.command()
    async def upscale(self, ctx, *args):
        message = ctx.message
        args = list(args)
        # Grab image URL
        if message.attachments:
            url = message.attachments[0].url
            args.insert(0, url)
            filename = message.attachments[0].filename
        else:
            try:
                url = args[0]
                filename = url[url.rfind("/")+1:]
            except:
                await message.channel.send('{}, you need to provide a url or an image attachment.'.format(message.author.mention))

        # Grab model name
        try:
            model_jobs = args[1].split(';')[:3]
        except:
            await message.channel.send('{}, you need to provide a model.'.format(message.author.mention))
            return

        try:
            downscale, filter, montage, blur_type, blur_amt, fixhist, seamless = self.parse_flags(
                args, message)
        except ValueError as e:
            await message.channel.send(e)
            return

        try:
            image = self.download_img_from_url(url)
        except:
            await message.channel.send('{}, your image could not be downloaded.'.format(message.author.mention))
            return

        if downscale:
            try:
                image = self.downscale_img(image, filter, downscale)
            except:
                await message.channel.send('{}, your image could not be downscaled.'.format(message.author.mention))
        if blur_type:
            try:
                if not blur_amt:
                    await message.channel.send('Unknown blur amount {}.'.format(blur_amt))
                elif type(blur_amt) != int:
                    await message.channel.send('Blur amount {} not a number.'.format(blur_amt))
                else:
                    # if blur_type in {'box', 'average'}:
                    #     image = cv2.boxFilter(
                    #         image, blur_amt, (blur_amt, blur_amt))
                    if blur_type in {'gauss', 'gaussian'}:
                        if blur_amt % 2 != 1:
                            blur_amt += 1
                        image = cv2.GaussianBlur(
                            image, (blur_amt, blur_amt), cv2.BORDER_DEFAULT)
                    elif blur_type in {'median'}:
                        if blur_amt % 2 != 1:
                            blur_amt += 1
                        image = cv2.medianBlur(image, blur_amt)
                    else:
                        await message.channel.send('Unknown blur type {}.'.format(blur_type))
            except:
                await message.channel.send('{}, your image could not be blurred.'.format(message.author.mention))
        if not image.shape[0] > config['img_size_cutoff'] and not image.shape[1] > config['img_size_cutoff']:
            if len(self.queue) == 0:
                self.queue[0] = {
                    'jobs': []
                }
                for model_job in model_jobs:
                    models = [self.aliases[process.extractOne(model.replace('.pth', ''), self.fuzzymodels)[
                        0]] for model in model_job.split('>')[:3]]
                    self.queue[0]['jobs'].append(
                        {'message': message, 'filename': filename, 'models': models, 'image': image})
                while (len(self.queue[0]['jobs']) > 0):
                    try:
                        job = self.queue[0]['jobs'].pop(0)
                        sent_message = await message.channel.send(f"{job['filename']} is being upscaled using {', '.join(job['models']) if len(job['models']) > 1 else job['models'][0]}")

                        img = job['image']

                        # this is needed for montaging with chains
                        og_image = img

                        for i in range(len(job['models'])):

                            img_height, img_width, img_channels = img.shape
                            dim = config['split_threshold']
                            overlap = 16

                            if not image.shape[0] > config['img_size_cutoff'] and not image.shape[1] > config['img_size_cutoff']:

                                # For some reason if either dim of the image is a multiple (or close) of the split size it crashes
                                # So, I just keep increasing the split size until its an acceptable number
                                # TODO: Figure out why it crashes in the first place
                                if img_height > 16 and img_width > 16:
                                    while img_height % dim < 16 or img_width % dim < 16:
                                        dim -= 16

                                do_split = img_height > dim or img_width > dim
                                await sent_message.edit(content=sent_message.content + ' | ')

                                if seamless:
                                    img = self.make_seamless(img)

                                if do_split:
                                    await sent_message.edit(content=sent_message.content + ' Splitting...')
                                    imgs, num_horiz, num_vert = self.split(
                                        img, dim, overlap)
                                else:
                                    imgs = [img]

                                await sent_message.edit(content=sent_message.content + ' Upscaling...')
                                rlts, scale = self.esrgan(
                                    imgs, job['models'][i])

                                if do_split:
                                    await sent_message.edit(content=sent_message.content + ' Merging...')
                                    rlt = self.merge(rlts, scale, overlap,
                                                     img_height, img_width, img_channels, num_horiz, num_vert)
                                else:
                                    rlt = rlts[0]

                                if seamless:
                                    rlt = self.crop_seamless(rlt, scale)

                                # attempts to fix broken alpha contrast caused by model
                                # if fixhist and img.ndim == 3 and img.shape[2] == 4:
                                #     # a = cv2.equalizeHist(a.astype('uint8'))
                                #     a = self.hist_match(
                                #         cv2.split(rlt)[3], cv2.split(img)[3])
                                #     rlt[:, :, 3] = a
                            else:
                                await message.channel.send('Unable to continue chain due to size cutoff ({}).'.format(config['img_size_cutoff']))
                                break

                            if len(models) > 1:
                                img = rlt.astype('uint8')

                        await sent_message.edit(content=sent_message.content + ' Sending...')

                        # converts result image to png bytestream
                        ext = '.png'
                        data = BytesIO(cv2.imencode('.png', rlt, [
                            cv2.IMWRITE_PNG_COMPRESSION, 16])[1].tostring())
                        if (len(data.getvalue()) >= 8000000):
                            ext = '.webp'
                            data = BytesIO(cv2.imencode('.webp', rlt, [
                                cv2.IMWRITE_WEBP_QUALITY, 64])[1].tostring())
                        # send result through discord
                        await job['message'].channel.send('{}, your image has been upscaled using {}.'.format(job['message'].author.mention, ', '.join(job['models']) if len(job['models']) > 1 else job['models'][0]), file=discord.File(data, job['filename'].split('.')[0] + ext))
                        await sent_message.edit(content=sent_message.content + ' Done.')
                    except Exception as e:
                        print(e)
                        await job['message'].channel.send('{}, there was an error upscaling your image.'.format(job['message'].author.mention))

                    if montage:
                        try:
                            montage_img = self.make_montage(
                                og_image, rlt, ', '.join(job['models']) if len(job['models']) > 1 else job['models'][0])
                            # converts result image to png bytestream
                            ext = '.png'
                            data = BytesIO(cv2.imencode('.png', montage_img, [
                                cv2.IMWRITE_PNG_COMPRESSION, 16])[1].tostring())
                            if (len(data.getvalue()) >= 8000000):
                                ext = '.webp'
                                data = BytesIO(cv2.imencode('.webp', montage_img, [
                                    cv2.IMWRITE_WEBP_QUALITY, 64])[1].tostring())
                            await job['message'].channel.send('{}, your montage has been created.'.format(job['message'].author.mention), file=discord.File(data, job['filename'].split('.')[0] + '_montage' + ext))
                        except:
                            await job['message'].channel.send('{}, there was an error creating your montage.'.format(job['message'].author.mention))
                self.queue.pop(0)
            else:
                for model_job in model_jobs:
                    models = [self.aliases[process.extractOne(model.replace('.pth', ''), self.fuzzymodels)[
                        0]] for model in model_job.split('>')[:3]]
                    self.queue[0]['jobs'].append(
                        {'message': message, 'filename': filename, 'models': models, 'image': image})
                    await message.channel.send('{}, {} has been added to the queue. Your image is #{} in line for processing.'.format(message.author.mention, filename, len(self.queue[0]['jobs'])))
        else:
            await message.channel.send('{}, your image is larger than the size threshold ({}).'.format(message.author.mention, config['img_size_cutoff']))

    def split(self, img, dim, overlap):
        '''
        Creates an array of equal length image chunks to use for upscaling

                Parameters:
                        img (array): Numpy image array 
                        dim (int): Number to use for length and height of image chunks
                        overlap (int): The amount of overlap between chunks

                Returns:
                        imgs (array): Array of numpy image "chunks"
                        num_horiz (int): Number of horizontal chunks
                        num_vert (int): Number of vertical chunks
        '''
        img_height, img_width, img_channels = img.shape
        num_horiz = math.ceil(img_width / dim)
        num_vert = math.ceil(img_height / dim)
        imgs = []
        for i in range(num_vert):
            for j in range(num_horiz):
                tile = img[i * dim:i * dim + dim + overlap,
                           j * dim:j * dim + dim + overlap].copy()
                imgs.append(tile)
        return imgs, num_horiz, num_vert

    # This method is a somewhat modified version of BlueAmulet's original pymerge script that is able to use my split chunks
    def merge(self, rlts, scale, overlap, img_height, img_width, img_channels, num_horiz, num_vert):
        '''
        Merges the image chunks back together

                Parameters:
                        rlts (array): The resulting images from ESRGAN 
                        scale (int): The scale of the model that was applied
                        overlap (int): The amount of overlap between chunks
                        img_height (int): The height of the original image
                        img_width (int): The width of the original image
                        num_horiz (int): Number of horizontal chunks
                        num_vert (int): Number of vertical chunks

                Returns:
                        rlt (array): Numpy image array of the resulting merged image
        '''
        rlt_overlap = int(overlap * scale)

        rlts_fin = [[None for x in range(num_horiz)]
                    for y in range(num_vert)]

        rlt = np.zeros((img_height * scale,
                        img_width * scale, img_channels))

        c = 0
        for tY in range(num_vert):
            for tX in range(num_horiz):
                img = rlts[tY*num_horiz+tX]
                shape = img.shape
                c = max(c, shape[2])
                rlts_fin[tY][tX] = img

        for tY in range(num_vert):
            for tX in range(num_horiz):
                img = rlts_fin[tY][tX]
                if img.shape[2] == 3 and c == 4:  # pad with solid alpha channel
                    img = np.dstack((img, np.full(img.shape[:-1], 1.)))
                    rlts_fin[tY][tX] = img
                shape = img.shape
                # Fade out edges
                # Left edge
                if tX > 0:
                    for x in range(rlt_overlap):
                        img[:, x] *= ((x + 1)/(rlt_overlap + 1))
                # Top edge
                if tY > 0:
                    for y in range(rlt_overlap):
                        img[y, :] *= ((y + 1)/(rlt_overlap + 1))
                # Right edge
                if tX < num_horiz - 1:
                    for x in range(rlt_overlap):
                        iX = x + shape[1] - rlt_overlap
                        img[:, iX] *= ((rlt_overlap - x) /
                                       (rlt_overlap + 1))
                # Bottom edge
                if tY < num_vert - 1:
                    for y in range(rlt_overlap):
                        iY = y + shape[0] - rlt_overlap
                        img[iY, :] *= ((rlt_overlap - y) /
                                       (rlt_overlap + 1))

        baseY = 0
        for tY in range(num_vert):
            baseX = 0
            for tX in range(num_horiz):
                img = rlts_fin[tY][tX]
                shape = img.shape

                # Copy non overlapping image data
                x1 = (0 if tX == 0 else rlt_overlap)
                y1 = (0 if tY == 0 else rlt_overlap)
                x2 = shape[1]
                y2 = shape[0]
                rlt[baseY+y1:baseY+y2, baseX +
                    x1:baseX+x2] = img[y1:y2, x1:x2]

                # Blend left
                if tX > 0:
                    rlt[baseY+y1:baseY+y2, baseX:baseX +
                        rlt_overlap] += img[y1:y2, :rlt_overlap]

                # Blend up
                if tY > 0:
                    rlt[baseY:baseY+rlt_overlap, baseX +
                        x1:baseX+x2] += img[:rlt_overlap, x1:x2]

                # Blend corner
                if tX > 0 and tY > 0:
                    rlt[baseY:baseY+rlt_overlap, baseX:baseX +
                        rlt_overlap] += img[:rlt_overlap, :rlt_overlap]

                baseX += shape[1] - rlt_overlap
            baseY += shape[0] - rlt_overlap
        return rlt

    # This code is a somewhat modified version of BlueAmulet's fork of ESRGAN by Xinntao
    def process(self, img):
        '''
        Does the processing part of ESRGAN. This method only exists because the same block of code needs to be ran twice for images with transparency.

                Parameters:
                        img (array): The image to process

                Returns:
                        rlt (array): The processed image
        '''
        if img.shape[2] == 3:
            img = img[:, :, [2, 1, 0]]
        elif img.shape[2] == 4:
            img = img[:, :, [2, 1, 0, 3]]
        img = torch.from_numpy(np.transpose(img, (2, 0, 1))).float()
        img_LR = img.unsqueeze(0)
        img_LR = img_LR.to(self.device)

        output = self.model(img_LR).data.squeeze(
            0).float().cpu().clamp_(0, 1).numpy()
        if output.shape[0] == 3:
            output = output[[2, 1, 0], :, :]
        elif output.shape[0] == 4:
            output = output[[2, 1, 0, 3], :, :]
        output = np.transpose(output, (1, 2, 0))
        return output

    # This code is a somewhat modified version of BlueAmulet's fork of ESRGAN by Xinntao
    def esrgan(self, imgs, model_name):
        '''
        Runs ESRGAN on all the images passed in with the specified model

                Parameters:
                        imgs (array): The images to run ESRGAN on
                        model_name (string): The model to use

                Returns:
                        rlts (array): The processed images
        '''
        model_path = './models/' + model_name
        # torch.device('cpu' if args.cpu else 'cuda')

        if model_path != self.last_model:
            state_dict = torch.load(model_path)

            if 'conv_first.weight' in state_dict:
                print('Error: Attempted to load a new-format model')
                sys.exit(1)

            # extract model information
            scale2 = 0
            max_part = 0
            if 'f_HR_conv1.0.weight' in state_dict:
                kind = 'SPSR'
                scalemin = 4
            else:
                kind = 'ESRGAN'
                scalemin = 6
            for part in list(state_dict):
                parts = part.split('.')
                n_parts = len(parts)
                if n_parts == 5 and parts[2] == 'sub':
                    nb = int(parts[3])
                elif n_parts == 3:
                    part_num = int(parts[1])
                    if part_num > scalemin and parts[0] == 'model' and parts[2] == 'weight':
                        scale2 += 1
                    if part_num > max_part:
                        max_part = part_num
                        out_nc = state_dict[part].shape[0]
            upscale = 2 ** scale2
            in_nc = state_dict['model.0.weight'].shape[1]
            if kind == 'SPSR':
                out_nc = state_dict['f_HR_conv1.0.weight'].shape[0]
            nf = state_dict['model.0.weight'].shape[0]

            if in_nc != self.last_in_nc or out_nc != self.last_out_nc or nf != self.last_nf or nb != self.last_nb or upscale != self.last_scale:
                if kind == 'ESRGAN':
                    self.model = arch.RRDB_Net(in_nc, out_nc, nf, nb, gc=32, upscale=upscale, norm_type=None, act_type='leakyrelu',
                                        mode='CNA', res_scale=1, upsample_mode='upconv')
                elif kind == 'SPSR':
                    self.model = arch.SPSRNet(in_nc, out_nc, nf, nb, gc=32, upscale=upscale, norm_type=None, act_type='leakyrelu',
                                     mode='CNA', upsample_mode='upconv')
                self.last_in_nc = in_nc
                self.last_out_nc = out_nc
                self.last_nf = nf
                self.last_nb = nb
                self.last_scale = upscale

            self.model.load_state_dict(state_dict, strict=True)
            del state_dict
            self.model.eval()
            for k, v in self.model.named_parameters():
                v.requires_grad = False
            self.model = self.model.to(self.device)

        rlts = []
        for img in imgs:
            # read image
            img = img * 1. / np.iinfo(img.dtype).max

            if img.ndim == 3 and img.shape[2] == 4 and self.last_in_nc == 3 and self.last_out_nc == 3:
                shape = img.shape
                # img1 = np.copy(img[:, :, :3])
                # img2 = np.copy(img[:, :, :3])
                # for c in range(3):
                #     img1[:, :, c] *= img[:, :, 3]
                #     img2[:, :, c] = (img2[:, :, c] - 1) * img[:, :, 3] + 1

                # output1 = self.process(img1)
                # output2 = self.process(img2)
                # alpha = 1 - np.mean(output2-output1, axis=2)
                # output = np.dstack((output1, alpha))
                # shape = output1.shape
                # divalpha = np.where(alpha < 1. / 510., 1, alpha)
                # for c in range(shape[2]):
                #     output[:, :, c] /= divalpha

                img1 = np.copy(img[:, :, :3])
                img2 = cv2.merge((img[:, :, 3], img[:, :, 3], img[:, :, 3]))
                output1 = self.process(img1)
                output2 = self.process(img2)
                output = cv2.merge(
                    (output1[:, :, 0], output1[:, :, 1], output1[:, :, 2], output2[:, :, 0]))
            else:
                if img.ndim == 2:
                    img = np.tile(np.expand_dims(img, axis=2),
                                  (1, 1, min(self.last_in_nc, 3)))
                if img.shape[2] > self.last_in_nc:  # remove extra channels
                    print('Warning: Truncating image channels')
                    img = img[:, :, :self.last_in_nc]
                # pad with solid alpha channel
                elif img.shape[2] == 3 and self.last_in_nc == 4:
                    img = np.dstack((img, np.full(img.shape[:-1], 1.)))
                output = self.process(img)

            output = (output * 255.0).round()
            # if output.ndim == 3 and output.shape[2] == 4:

            rlts.append(output)
        torch.cuda.empty_cache()
        return rlts, upscale

    # Method translated to python from BlueAmulet's original alias PR
    # Basically this allows the fuzzy matching to look at individual phrases present in the model name
    # This way, if you had two models, e.g 4xBox and 4x_sponge_bob, you could type 'bob' and it will choose the correct model
    # This method just builds the alias dictionary and list for that functionality
    def build_aliases(self):
        '''Builds aliases for fuzzy string matching the model name input'''
        aliases = {}

        # Create aliases for models based on unique parts
        for model in self.models:
            name = os.path.splitext(os.path.basename(model))[0]
            parts = re.findall(
                r'([0-9]+x?|[A-Z]+(?![a-z])|[A-Z][^A-Z0-9_-]*)', name)
            for i in range(len(parts)):
                for j in range(i+1, len(parts)+1):
                    alias = ''.join(parts[i:j])
                    if alias in aliases:
                        if fuzz.ratio(alias, model) > fuzz.ratio(alias, aliases[alias]):
                            aliases[alias] = model
                    else:
                        aliases[alias] = model

        # Ensure exact names are usable
        for model in self.models:
            name = os.path.splitext(os.path.basename(model))[0]
            aliases[name] = model

        # Build list of usable aliases
        fuzzylist = []
        for alias in aliases:
            if aliases[alias]:
                fuzzylist.append(alias)

        print('Made {} aliases for {} models.'.format(
            len(fuzzylist), len(self.models)))
        return fuzzylist, aliases

    def make_montage(self, img1, img2, label):
        '''Creates a side-by-side comparison of two images with a label

                Parameters:
                        img1 (array): The left image
                        img2 (array): The right image
                        label (string): The label to apply underneath (usually model name)
                Returns:
                        img (array): The montaged image
        '''
        img1 = cv2.resize(
            img1, (img2.shape[1], img2.shape[0]), interpolation=cv2.INTER_NEAREST)
        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
        img = cv2.hconcat([img1, img2])
        img = cv2.copyMakeBorder(
            img,
            0, 36, 0, 0,
            cv2.BORDER_CONSTANT,
            value=(0, 0, 0, 255)
        )

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_size = 1
        font_thickness = 2

        textsize = cv2.getTextSize(label, font, font_size, font_thickness)[0]
        textX = math.floor((img.shape[1] - textsize[0]) / 2)
        textY = math.ceil(img.shape[0] - ((40 - textsize[1]) / 2))

        cv2.putText(img, label, (textX, textY), font, font_size,
                    color=(255, 255, 255, 255), thickness=font_thickness, lineType=cv2.LINE_AA)
        return img

    # This histogram matching code is from https://stackoverflow.com/questions/32655686/histogram-matching-of-two-images-in-python-2-x
    def hist_match(self, source, template):
        olddtype = source.dtype
        oldshape = source.shape
        source = source.ravel()
        template = template.ravel()

        s_values, bin_idx, s_counts = np.unique(source, return_inverse=True,
                                                return_counts=True)
        t_values, t_counts = np.unique(template, return_counts=True)
        s_quantiles = np.cumsum(s_counts).astype(np.float64)
        s_quantiles /= s_quantiles[-1]
        t_quantiles = np.cumsum(t_counts).astype(np.float64)
        t_quantiles /= t_quantiles[-1]
        interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)
        interp_t_values = interp_t_values.astype(olddtype)

        return interp_t_values[bin_idx].reshape(oldshape)

    def make_seamless(self, img):
        img_height, img_width, img_channels = img.shape
        img = cv2.hconcat([img, img, img])
        img = cv2.vconcat([img, img, img])
        y, x = img_height - 16, img_width - 16
        h, w = img_height + 32, img_width + 32
        img = img[y:y+h, x:x+w]
        return img

    def crop_seamless(self, img, scale):
        img_height, img_width, img_channels = img.shape
        y, x = 16 * scale, 16 * scale
        h, w = img_height - (32 * scale), img_width - (32 * scale)
        img = img[y:y+h, x:x+w]
        return img

    def parse_flags(self, args, message):
        downscale = None
        filter = cv2.INTER_AREA
        montage = False
        blur_type = None
        blur_amt = None
        fixhist = False
        seamless = False
        for index, arg in enumerate(args):
            arg = arg.lower()
            if arg in {'-downscale', '-d'}:
                try:
                    downscale = float(args[index + 1])
                except:
                    raise ValueError('Downscale value not specified.')
            elif arg in {'-filter', '-f'}:
                try:
                    interpolation = args[index + 1].lower()
                    if interpolation in {'point', 'nearest', 'nn', 'nearest_neighbor', 'nearestneighbor'}:
                        filter = cv2.INTER_NEAREST
                    elif interpolation in {'box', 'area'}:
                        filter = cv2.INTER_AREA
                    elif interpolation in {'linear', 'bilinear', 'triangle'}:
                        filter = cv2.INTER_LINEAR
                    elif interpolation in {'cubic', 'bicubic'}:
                        filter = cv2.INTER_CUBIC
                    elif interpolation in {'exact', 'linear_exact', 'linearexact'}:
                        filter = cv2.INTER_LINEAR_EXACT
                    elif interpolation in {'lanczos', 'lanczos4'}:
                        filter = cv2.INTER_LANCZOS4
                    else:
                        raise ValueError(
                            'Unknown image filter {}.'.format(interpolation))
                except:
                    raise ValueError('Filter value not specified.')
            elif arg in {'-blur', '-b'}:
                try:
                    blur_type = args[index + 1]
                    blur_amt = int(args[index + 2])
                except:
                    raise ValueError(
                        'Blur requires 2 arguments, type and amount.')
            elif arg in {'-montage', '-m'}:
                montage = True
            elif arg in {'-fixhist', '-fh', '-fix'}:
                fixhist = True
            elif arg in {'-seamless', '-s'}:
                seamless = True
        return downscale, filter, montage, blur_type, blur_amt, fixhist, seamless

    def downscale_img(self, image, filter, amt):
        scale_percent = 1 / float(amt) * 100
        width = int(image.shape[1] * scale_percent / 100)
        height = int(image.shape[0] * scale_percent / 100)
        dim = (width, height)
        image = cv2.resize(image, dim, interpolation=filter)
        return image

    def download_img_from_url(self, img):
        req = urllib.request.Request(img, headers={
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/35.0.1916.47 Safari/537.36'
        })
        url = urllib.request.urlopen(req)
        image = np.asarray(bytearray(url.read()), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_UNCHANGED)
        return image


bot.add_cog(ESRGAN(bot))

bot.run(config['bot_token'])
