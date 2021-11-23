const fs = require('fs');

let trainedNet;
try {
  trainedNet = JSON.parse(fs.readFileSync('network.json'));
} catch (e) {}
process.stdin.resume();
process.on('SIGINT', () => {
  mapNetwork();
  console.log('SIGINT - Saving network');
  fs.writeFileSync('network.json', JSON.stringify(trainedNet));
  process.exit();
});

const dataFileBuffer = fs.readFileSync('./dataset/train-images-idx3-ubyte');
const labelFileBuffer = fs.readFileSync('./dataset/train-labels-idx1-ubyte');
const pixelValues = [];
for (let image = 0; image < 60000; image++) {
  const pixels = [];
  for (let x = 0; x < 28; x++) {
    for (let y = 0; y < 28; y++) {
      pixels.push(dataFileBuffer[image * 28 * 28 + (x + y * 28) + 15]);
    }
  }
  pixelValues.push({ value: Number(JSON.stringify(labelFileBuffer[image + 8])), pixels });
}

// NETWORK INITIALIZATION:
const layers = [784, 392, 196, 98, 10];
const network = [];
let neuronId = 0;
let weightId = 0;
const getRandom = () => Math.random() * 0.4 - 0.2; // brain.js:1413
layers.forEach((neuronsCountCurrentLayer, index) => {
  const neuronsCountPrevLayer = layers[index - 1];
  const neuronsCountNextLayer = layers[index + 1];
  const layer = [];
  for (let i = 0; i < neuronsCountCurrentLayer; i++) {
    const neuron = { input: [], output: [], id: neuronId++, layerId: index };
    if (index > 0) {
      neuron.value = neuron.delta = neuron.error = 0;
      neuron.bias = trainedNet ? trainedNet[index][i].bias : getRandom();
    }
    if (neuronsCountPrevLayer) {
      network[index - 1].forEach(neuronPrev => {
        neuronPrev.output.forEach((weight, weightIndex) => {
          i === weightIndex && neuron.input.push(weight);
        });
      });
    }
    if (neuronsCountNextLayer) {
      for (let j = 0; j < neuronsCountNextLayer; j++) {
        const weight = { change: 0, id: weightId++ };
        weight.value = trainedNet ? trainedNet[index][i].weights[j] : getRandom();
        neuron.output.push(weight);
      }
    }
    layer.push(neuron);
  }
  network.push(layer);
});

const round = x => Math.round(x * 10000) / 10000; // 0.4115000143.. -> 0.4115
const joinString = i => i.join(' | ').slice(0, 90);
const weightString = w => `id:${w.id} value:${round(w.value)} change:${round(w.change)}`;
const layerMap = neuron => ({
  id: neuron.id,
  layer: `L${neuron.layerId}`,
  ['input weights']: joinString(neuron.input.map(weightString)),
  value: round(neuron.value),
  bias: typeof neuron.bias === 'number' ? round(neuron.bias) : '',
  error: typeof neuron.error === 'number' ? round(neuron.error) : '',
  delta: typeof neuron.delta === 'number' ? round(neuron.delta) : '',
  ['output weights']: joinString(neuron.output.map(weightString)),
});
const mapNetwork = () => {
  trainedNet = network.map((layer, layerIndex) => {
    return layer.map(n => {
      const neuronObj = {};
      if (layerIndex > 0) {
        neuronObj.bias = n.bias;
      }
      if (layerIndex < network.length - 1) {
        neuronObj.weights = n.output.map(w => w.value);
      }
      return neuronObj;
    });
  });
};
const printNetwork = () => {
  console.table(network.flat(1).map(layerMap));
  mapNetwork();
  console.log(`trainedNet JSON:`, JSON.stringify(trainedNet));
};

// printNetwork();debugger;

const sigmoid = x => 1 / (1 + Math.exp(-x));
const learnRate = 0.2;
const momentum = 0.1;
const run = (input, expected = []) => {
  // SET INITIAL VALUES:
  network[0].forEach((neuron, index) => (neuron.value = input[index])); // set input

  // PROJECT VALUES:
  network.forEach((layer, layerIndex) => {
    const nextLayer = network[layerIndex + 1] || [];
    layer.forEach((neuron, neuronIndex) => {
      neuron.output.forEach((weight, weightIndex) => {
        nextLayer[weightIndex].value += weight.value * neuron.value;
      });
    });
    nextLayer.forEach((neuron, neuronIndex) => (neuron.value = sigmoid(neuron.value + neuron.bias)));
  });

  // TRAIN:
  if (expected.length) {
    // CALCULATE DELTAS:
    for (let layerIndex = network.length - 1; layerIndex > 0; layerIndex--) {
      const layer = network[layerIndex];
      const nextLayer = network[layerIndex + 1] || [];
      layer.forEach((neuron, neuronIndex) => {
        if (layerIndex === network.length - 1) {
          neuron.error = expected[neuronIndex] - neuron.value;
        } else {
          neuron.error = 0;
          nextLayer.forEach(neuronNext => {
            neuron.error += neuronNext.delta * neuronNext.input[neuronIndex].value;
          });
        }
        neuron.delta = neuron.error * neuron.value * (1 - neuron.value);
      });
    }

    // ADJUST WEIGHTS:
    for (let layerIndex = 1; layerIndex < network.length; layerIndex++) {
      const layer = network[layerIndex];
      const prevLayer = network[layerIndex - 1] || [];
      layer.forEach(neuron => {
        neuron.input.forEach((weight, weightIndex) => {
          weight.change = learnRate * neuron.delta * prevLayer[weightIndex].value + momentum * weight.change;
          weight.value += weight.change;
        });
        neuron.bias += learnRate * neuron.delta;
      });
    }
  }

  return network[network.length - 1].map(i => i.value); // last layer values
};

/*

var z = [
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,243,203,13,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,116,253,149,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,242,253,149,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,248,253,96,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,49,250,253,19,0,0,0,0,0,0,0,0,0,0,0,61,99,43,0,0,0,0,0,0,0,0,0,0,125,253,253,19,0,0,0,0,0,24,189,110,0,0,12,191,253,229,42,0,0,0,0,0,0,0,0,0,125,253,159,3,0,0,0,0,0,176,253,245,66,0,33,253,253,253,227,0,0,0,0,0,0,0,0,0,125,253,142,0,0,0,0,0,60,243,253,253,78,0,6,45,128,253,252,102,0,0,0,0,0,0,0,0,125,253,142,0,0,0,0,0,176,253,253,211,16,0,0,0,4,165,253,159,0,0,0,0,0,0,0,0,125,253,142,0,0,0,0,61,242,253,253,201,0,0,0,0,0,143,253,253,0,0,0,0,0,0,0,0,154,253,142,0,0,0,51,186,253,253,253,171,0,0,0,0,0,77,253,253,0,0,0,0,0,0,0,0,255,253,142,0,0,0,215,253,201,118,253,98,0,0,0,0,0,143,253,253,0,0,0,0,0,0,0,0,164,253,219,13,18,160,242,253,91,66,253,237,54,0,0,0,0,143,253,230,0,0,0,0,0,0,0,0,106,252,253,176,202,253,253,241,44,56,243,253,171,13,0,0,4,160,253,123,0,0,0,0,0,0,0,0,0,149,252,253,253,253,242,86,0,0,172,253,253,101,0,0,7,182,253,123,0,0,0,0,0,0,0,0,0,0,93,97,97,97,65,0,0,0,23,196,253,238,136,0,13,211,253,123,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,49,247,253,240,136,38,253,253,123,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,169,253,253,253,253,253,249,54,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,21,136,251,253,253,253,176,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,80,123,200,179,57,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0] // 3

 [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,6,117,59,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,33,170,254,196,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,5,172,254,254,126,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,155,254,254,248,47,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,27,161,254,254,254,210,0,0,0,0,0,0,0,0,0,0,18,89,117,15,0,0,0,0,0,0,0,27,160,254,213,100,254,151,0,0,0,0,0,0,0,0,0,18,169,242,183,23,0,0,0,0,0,0,5,162,254,218,23,124,254,68,0,0,0,0,0,0,0,0,0,77,254,157,0,0,0,0,0,0,0,0,113,254,224,18,0,83,254,109,0,0,0,0,0,0,0,0,0,119,254,42,0,0,0,0,0,0,6,85,244,240,45,0,0,52,254,140,0,0,0,0,0,0,0,0,0,203,227,15,0,0,0,0,0,11,153,254,233,61,0,0,0,136,254,62,0,0,0,0,0,0,0,0,0,255,191,0,0,0,0,0,30,186,253,196,16,0,0,0,0,219,234,18,0,0,0,0,0,0,0,0,0,254,131,0,0,0,0,56,251,254,122,0,0,0,0,0,0,219,227,0,0,0,0,0,0,0,0,0,0,255,126,0,0,62,191,251,235,52,1,0,0,0,0,0,34,240,172,0,0,0,0,0,0,0,0,0,0,173,248,232,232,249,252,170,17,0,0,0,0,0,0,0,54,254,56,0,0,0,0,0,0,0,0,0,0,23,142,249,220,166,34,0,0,0,0,0,0,0,0,0,8,35,7,0,0,0,0,0,0,0,0,0,0,0,0,44,8,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0] // 2

 [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,104,153,80,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,22,147,254,253,230,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,22,178,253,254,253,221,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,114,253,253,220,253,104,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,26,70,99,0,0,19,237,180,84,255,216,34,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,81,214,245,249,230,197,204,253,243,239,254,98,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,123,245,244,205,192,244,253,254,253,242,246,113,4,0,0,0,0,0,0,0,0,0,0,0,0,0,0,85,247,253,94,0,13,52,228,254,185,38,50,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,85,221,254,155,25,0,0,49,254,236,50,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,22,247,254,227,21,0,0,0,166,253,134,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,66,253,254,89,0,0,0,22,228,210,17,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,182,253,169,6,0,0,0,181,253,94,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,47,254,254,93,0,0,0,95,254,205,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,97,253,202,9,0,0,68,245,253,121,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,138,253,128,15,32,116,215,247,128,9,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,138,253,194,195,228,253,219,118,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,231,254,254,255,254,197,50,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,146,253,253,140,56,4,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,138,253,253,93,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,89,194,94,25,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0] // 8

var resultHTML = '', id = 0;
for(let i = 0; i<28; i++){
  for (let j = 0; j<28;j++) {
    resultHTML += `<div style="width:26px;height:26px;display:inline-block;background-color:rgb( ${z[i+j*28]} 0 0);color:white;">${id++}</div>`;
  }
  resultHTML += '<br>'
}
document.body.innerHTML = resultHTML;0;

*/

// for (let i = 0; i < 20; i++) {
//   const testId = 4013 + i;
//   const result = run(pixelValues[testId].pixels);
//   console.log(pixelValues[testId].value, result.map((i, index) => `${index} : ${round(i)}`).join('\t'));
// }
// const testId = 4006;
// console.log(testId, pixelValues[testId].value, JSON.stringify(pixelValues[testId].pixels));
// process.exit();

const run2 = async () => {
  let i = 0;
  while (true) {
    i++;
    await new Promise(resolve => setTimeout(resolve, 1)); // delay to allow Ctrl+C interruption
    const id = Math.floor(Math.random() * pixelValues.length);
    const expected = Array(10)
      .fill(0)
      .map((j, index) => (pixelValues[id].value === index ? 1 : 0));
    run(pixelValues[id].pixels, expected); // train
    if (i % 100 === 0) {
      const testId2 = 4000; // 7
      const testId3 = 4006; // 8
      const result = run(pixelValues[testId2].pixels);
      const result2 = run(pixelValues[testId3].pixels);
      if (result[7] > 0.9 && result2[8] > 0.9) {
        console.log(`iteration: ${i}\t\t - reached 90% level, training stopped`);
        fs.writeFileSync('network.json', JSON.stringify(trainedNet));
        process.exit();
        // break;
      } else {
        console.log(
          `iteration: ${i}\t\t`,
          pixelValues[testId2].value,
          result.map((i, index) => `${index} : ${round(i)}`)[7], //.join('\t'),
          pixelValues[testId3].value,
          result2.map((i, index) => `${index} : ${round(i)}`)[8]
        );
      }
    }
  }
};
run2();

// printNetwork();