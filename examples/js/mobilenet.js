const fs = require('fs');
const Redis = require('ioredis');
const Jimp = require('jimp');
const Helpers = require('./Helpers');

// Simple configuration, all these files should exist already in the example folder
const config = {
   jsonClassIndex: 'imagenet_class_index.json',
   modelFile: '../models/mobilenet_v2_1.4_224_frozen.pb',
   inputPath: 'input',
   outputPath: 'MobilenetV2/Predictions/Reshape_1',
   imageHeight: 224,
   imageWidth: 224,
};

const answerSet = [];
const helpers  = new Helpers();
const labels   = JSON.parse(fs.readFileSync(config.jsonClassIndex));
const redis    = new Redis({ parser: 'javascript' });

let progress = 0;

function run(filenames) {

   /**
    * We're looping over each image and seeing if we can find a match
    */
   filenames.forEach((filename, key, filenames) => {
      Jimp.read(filename).then((inputImage) => {
         console.log(`\nWorking on ${filename}`);
         let image = inputImage.cover(config.imageWidth, config.imageHeight);
         let normalized = helpers.normalizeRGB(image.bitmap.data, image.hasAlpha());
         let buffer = Buffer.from(normalized.buffer);

         console.log("...Setting input tensor \n...running model");

         /**
          * Redis Pipelines, this allows us to set up and batch our commands
          * If you want to learn more about how IORedis does this check out
          * https://github.com/luin/ioredis#pipelining
          */
         redis.pipeline()
              .call('AI.TENSORSET', 'input_' + key, 'FLOAT', 1, config.imageWidth, config.imageHeight, 3, 'BLOB', buffer)
              .call('AI.MODELRUN', 'mobilenet', 'INPUTS', 'input_' + key, 'OUTPUTS', 'output_' + key)
              .callBuffer('AI.TENSORGET', 'output_' + key, 'BLOB')
              .exec((err, results) => {

                  /**
                   * THis looks a wierd, but this is because of how pipelines
                   * work. When the pipeline is executed the output is
                   * put into the results as part of an array, so
                   * in this case, we don't need the output
                   * from the first or second `.calls` we
                   * only need the data from the
                   * AI.TENSORGET() so this
                   * just helps us get to
                   * the right data
                   */
                  let bufferResults = results[2][1];

                  let label = helpers.argmax(
                           helpers.bufferToFloat32Array(bufferResults[bufferResults.length - 1]));

                  // We push our match into the answer set
                  answerSet.push({
                     'filename': filename,
                     'matches': labels[label-1][1]
                  });

                  /**
                   * forEach is not async, but the pipeline calls are so we need
                   * to set a progress check and wait until we've actually
                   * checked all the images. Once we know we've looked
                   * at all the images then we can display the table
                   * results. This stops us from having multiple
                   * tables for each image that gets processed
                   */
                  progress++;
                  if (progress === filenames.length) {
                     console.log("\n...OK I think I got something...\n");
                     console.table(answerSet, ['filename', 'matches']);
                  }
               });
      });
   });
}

run(Array.from(process.argv).splice(2));
