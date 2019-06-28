class Helpers {

   argmax(arr) {
      let index = 0;
      let value = arr[0];

      arr.forEach((item, key) => {
         if (item > value) {
            value = item;
            index = key;
         }
      });

      return index;
   }

   normalizeRGB(buffer) {
      let npixels = buffer.length / 4;
      let out = new Float32Array(npixels * 3);

      for (let i=0; i<npixels; i++) {
         out[3*i]   = buffer[4*i]   / 128 - 1;
         out[3*i+1] = buffer[4*i+1] / 128 - 1;
         out[3*i+2] = buffer[4*i+2] / 128 - 1;
      }

      return out;
   }

   bufferToFloat32Array(buffer) {
      let out_array = new Float32Array(buffer.length / 4);
      for (let i=0; i<out_array.length; i++) {
         out_array[i] = buffer.readFloatLE(4*i);
      }
      return out_array;
   }
}

module.exports = Helpers;
