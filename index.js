let trainedNet; // pre-trained network example, uncomment bellow:
// trainedNet = [
//   [
//     { bias: 0.03170031433221565, output: [-7.469626099371924] },
//     { bias: -0.0633157356352152, output: [7.332374145567543] },
//   ],
//   [{ bias: 0.22367691657209912, output: [-7.977731379883116, 7.9752563637802165] }],
//   [
//     { bias: 3.6543471429420475, output: [] },
//     { bias: -3.6531504158918713, output: [] },
//   ],
// ];

// NETWORK INITIALIZATION:
const layers = [2, 1, 2];
const network = [];
let neuronId = 0;
let weightId = 0;
layers.forEach((neuronsCountCurrentLayer, index) => {
  const neuronsCountPrevLayer = layers[index - 1];
  const neuronsCountNextLayer = layers[index + 1];
  const layer = [];
  for (let i = 0; i < neuronsCountCurrentLayer; i++) {
    const neuron = { input: [], output: [], delta: 0, error: 0, value: 0, id: neuronId++, layerId: index };
    neuron.bias = trainedNet ? trainedNet[index][i].bias : Math.random() * 0.4 - 0.2; // brain.js:1413
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
        weight.value = trainedNet ? trainedNet[index][i].output[j] : Math.random() * 0.4 - 0.2;
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
const printNetwork = () => {
  const layerMap = neuron => ({
    id: neuron.id,
    layer: `L${neuron.layerId}`,
    input: joinString(neuron.input.map(weightString)),
    value: round(neuron.value),
    bias: round(neuron.bias),
    delta: round(neuron.delta),
    error: round(neuron.error),
    output: joinString(neuron.output.map(weightString)),
  });
  console.table(network.flat(1).map(layerMap));
  const trainedNet = network.map(l => l.map(n => ({ bias: n.bias, output: n.output.map(w => w.value) })));
  console.log(`trainedNet JSON:`, JSON.stringify(trainedNet));
};

// printNetwork();debugger;

const sigmoid = x => 1 / (1 + Math.exp(-x));
const learnRate = 0.7;
const momentum = 0.1;
const run = (input, expected = []) => {
  // SET INITIAL VALUES:
  network[0].forEach((neuron, index) => (neuron.value = input[index])); // set input

  // PROJECT VALUES:
  network.forEach((layer, layerIndex) => {
    const nextLayer = network[layerIndex + 1] || [];
    layer.forEach((neuron, neuronIndex) => {
      neuron.output.forEach((weight, weightIndex) => {
        nextLayer[weightIndex].value = nextLayer[weightIndex].value + weight.value * neuron.value;
      });
    });
    nextLayer.forEach((neuron, neuronIndex) => {
      neuron.value = sigmoid(neuron.bias + neuron.value); // bias + sum
    });
  });

  // TRAIN:
  if (expected.length) {
    // CALCULATE DELTAS:
    for (let layerIndex = network.length - 1; layerIndex >= 0; layerIndex--) {
      const layer = network[layerIndex];
      const nextLayer = network[layerIndex + 1] || [];
      layer.forEach((neuron, neuronIndex) => {
        neuron.error = 0;
        if (layerIndex === network.length - 1) {
          neuron.error = expected[neuronIndex] - neuron.value;
        } else {
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
      layer.forEach((neuron, neuronIndex) => {
        neuron.input.forEach((weight, weightIndex) => {
          weight.change = learnRate * neuron.delta * prevLayer[weightIndex].value + momentum * weight.change;
          weight.value += weight.change;
        });
        neuron.bias += learnRate * neuron.delta;
      });
    }
  }

  return network[network.length - 1].map(i => i.value); // result - last layer values
};

for (let i = 0; i < 10000; i++) {
  const expected = [Math.floor(Math.random() * 2)];
  expected[1] = expected[0] === 0 ? 1 : 0;
  const input = expected.map(x => round((Math.random() + x) / 2));
  run(input, expected); // train
  if (i % 500 === 0) {
    const result = run([0.2, 0.6]); // expected [0, 1];
    if (result[0] < 0.02) {
      console.log(`iteration: ${i}\t\t`, result, `- reached 2% level, training stopped`);
      break;
    } else {
      console.log(`iteration: ${i}\t\t`, result);
    }
  }
}

printNetwork();
