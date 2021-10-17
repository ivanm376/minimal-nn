// const error = 0.05
const sigmoid = x => 1 / (1 + Math.exp(-x));
const learnRate = 0.5;
const momentum = 0.1;
const round = x => Math.round(x * 10000) / 10000; // 0.42851.. >> 0.43

let trainedNet;
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

const config = { layers: [2, 1, 2] };
const { layers } = config;
let network = [];
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
    error: round(neuron.delta),
    expected: neuron.expected, // expected value
    output: joinString(neuron.output.map(weightString)),
  });
  console.table(network.flat(1).map(layerMap));
  const trainedNet = network.map(l => l.map(n => ({ bias: n.bias, output: n.output.map(w => w.value) })));
  console.log(`trainedNet JSON:`, JSON.stringify(trainedNet));
};
// printNetwork();debugger;

const run = (input, expected = []) => {
  input = Float32Array.from(input);
  network.forEach((layer, layerIndex) => {
    // set initial values
    layer.forEach((neuron, index) => {
      neuron.value = layerIndex === 0 ? input[index] : 0;
      if (expected.length) {
        neuron.expected = layerIndex === network.length - 1 ? expected[index] : 0;
      } else {
        delete neuron.expected;
      }
    });
  });

  // PROJECT VALUES:
  network.forEach((layer, layerIndex) => {
    const nextLayer = network[layerIndex + 1] || [];
    layer.forEach((neuron, neuronIndex) => {
      neuron.output.forEach((weight, weightIndex) => {
        nextLayer[weightIndex].value = nextLayer[weightIndex].value + weight.value * neuron.value;
      });
    });
    nextLayer.forEach((neuron, neuronIndex) => {
      // console.log(neuron.bias === net.biases[layerIndex + 1][neuronIndex]); // - true
      neuron.value = sigmoid(neuron.bias + neuron.value); // bias + sum
    });
  });

  const result = network[network.length - 1].map(i => i.value);

  // TRAIN:
  if (expected.length) {
    // CALCULATE DELTAS:
    for (let layerIndex = network.length - 1; layerIndex >= 0; layerIndex--) {
      const layer = network[layerIndex];
      const nextLayer = network[layerIndex + 1] || [];
      layer.forEach((neuron, neuronIndex) => {
        let error = 0;
        // neuron.value = 0.46613582968711853;
        if (layerIndex === network.length - 1) {
          error = neuron.expected - neuron.value;
        } else {
          const deltas = nextLayer.map(i => i.delta); // this.deltas[layer + 1];
          for (let k = 0; k < deltas.length; k++) {
            error += deltas[k] * nextLayer[k].input[neuronIndex].value;
          }
        }

        // activeError[neuronIndex] = error; // neuron.error
        neuron.error = error;
        neuron.delta = error * neuron.value * (1 - neuron.value); // activeDeltas[neuronIndex]
      });
    }

    // ADJUST WEIGHTS:
    for (let layerIndex = 1; layerIndex < network.length; layerIndex++) {
      const layer = network[layerIndex];
      const prevLayer = network[layerIndex - 1] || [];
      const incoming = prevLayer.map(i => i.value);
      layer.forEach((neuron, neuronIndex) => {
        neuron.input.forEach((weight, weightIndex) => {
          let change = weight.change || 0; // activeChanges[node][k];
          change = learnRate * neuron.delta * incoming[weightIndex] + momentum * change;
          // console.log('a2', neuron.delta, incoming[weightIndex], weight.change, change, weight.value);
          weight.change = change;
          weight.value += change; // activeWeights[node][k]
        });
        neuron.bias += learnRate * neuron.delta;
      });
    }
  }

  return result;
};

// printNetwork();debugger;

const setExpected = x => round((Math.random() + x) / 2);
for (let i = 0; i < 10000; i++) {
  const expected = [Math.floor(Math.random() * 2)]; // config expected result
  expected[1] = expected[0] === 0 ? 1 : 0;
  run([setExpected(expected[0]), setExpected(expected[1])], expected);
  if (i % 500 === 0) {
    const result = run([0.2, 0.6], [0, 1]); // expected [0, 1];
    if (result[0] < 0.02) {
      console.log(`iteration: ${i}\t\t`, result, `- reached 2% level, training stopped`);
      break;
    } else {
      console.log(`iteration: ${i}\t\t`, result);
    }
  }
}

printNetwork();
// debugger;
