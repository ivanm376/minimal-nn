let weights = [0.42, 0.17, 0.78, 0.55, 0.22, 0.99, 0.12, 0.88]; // random weights
// const error = 0.05
const learnRate = 0.003; // 0.3 %
const round = x => Math.round(x * 10000) / 10000; // 0.42851.. >> 0.43
const sigmoid = x => round(1 / (1 + Math.exp(-x))); // range: 0 - 1

//  i1 - w1 - n1 - w5 - r1 (e1)
//     - w2 -    - w6 -
//  i2 - w3 - n2 - w7 - r2 (e2)
//     - w4 -    - w8 -
// i     w     r
// 0.6   1   > 1
// 0.6   0.5 > 0.6
// 0.6   0   > 0

const adjustWeight = (oldWeight, increase) => {
  if (increase) {
    return round(oldWeight + (1 - oldWeight) * learnRate);
  } else {
    return round(oldWeight - oldWeight * learnRate);
  }
};
const projectWeight = (input, weight) => {
  if (weight >= 0.5) {
    // weight === 1    > multiplier === 1
    // weight === 0.75 > multiplier === 0.5
    // weight === 0.5  > multiplier === 0
    return input + (1 - input) * (weight * 2 - 1); // result 0 - 1
  } else {
    // weight === 0.499 > multiplier === 0.9999 ~
    // weight === 0.25  > multiplier === 0.5
    // weight === 0     > multiplier === 0
    return input * weight * 2; // result 0 - 1
  }
};
const projectWeightBack = (result, weight) => {
  if (weight >= 0.5) {
    // 0.92 === 0.9 + (1 - 0.9) * (0.6 * 2 - 1)
    // 0.92 === x + (1 - x) * (0.6 * 2 - 1)
    // 0.92 === x + (1 - x) * 0.2
    // 0.92 === x + (1 - x) * 0.2
    const multiplier = weight * 2 - 1;
    return (result - multiplier) / (1 - multiplier);
  } else {
    return result / 2 / weight;
  }
};

const config = { layers: [2, 5, 5, 2] };
const { layers } = config;
let weightId = 0; // weights count
let neuronId = 0;
let network = [];
layers.forEach((neuronsCountCurrentLayer, index) => {
  const neuronsCountPrevLayer = layers[index - 1];
  const neuronsCountNextLayer = layers[index + 1];
  const layer = [];
  for (let i = 0; i < neuronsCountCurrentLayer; i++) {
    const neuron = { input: [], output: [], id: neuronId++, layerId: index };
    if (neuronsCountPrevLayer) {
      network[index - 1].forEach(neuronPrev => {
        neuronPrev.output.forEach((weight, weightIndex) => {
          i === weightIndex && neuron.input.push(weight);
        });
      });
    }
    if (neuronsCountNextLayer) {
      for (let i = 0; i < neuronsCountNextLayer; i++) {
        const weight = { value: round(Math.random()), id: weightId++ };
        neuron.output.push(weight);
      }
    }
    layer.push(neuron);
  }
  network.push(layer);
});
const printNetwork = () => {
  const layerMap = neuron => ({
    id: neuron.id,
    layer: `L${neuron.layerId}`,
    input: neuron.input.map(weight => `id:${weight.id} value:${weight.value}`).join(' | '),
    value: neuron.value,
    expected: neuron.expected, // expected value
    output: neuron.output.map(weight => `id:${weight.id} value:${weight.value}`).join(' | '),
  });
  console.table(network.reduce((a, b) => a.concat(b)).map(layerMap));
};
// printNetwork();debugger;

const run = (input, expected = []) => {
  // printNetwork();
  // debugger;
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
  // printNetwork();
  // debugger;

  // PROJECT VALUES:
  network.forEach((layer, layerIndex) => {
    const nextLayer = network[layerIndex + 1] || [];
    layer.forEach((neuron, neuronIndex) => {
      neuron.output.forEach((weight, weightIndex) => {
        // console.log(round(projectWeight(neuron.value, weight.value)));
        nextLayer[weightIndex].value += projectWeight(neuron.value, weight.value);
        nextLayer[weightIndex].value = round(nextLayer[weightIndex].value);
      });
    });
    nextLayer.forEach(neuron => (neuron.value = round(neuron.value / layer.length)));
  });

  // TRAIN:
  if (expected.length) {
    for (let layerIndex = network.length - 1; layerIndex >= 0; layerIndex--) {
      const layer = network[layerIndex];
      const prevLayer = network[layerIndex - 1] || [];
      // console.log(layerIndex, layer, prevLayer);
      layer.forEach((neuron, neuronIndex) => {
        neuron.input.forEach((weight, weightIndex) => {
          weight.value = adjustWeight(weight.value, neuron.value < neuron.expected);
          prevLayer[weightIndex].expected += round(projectWeightBack(neuron.value, weight.value));
          prevLayer[weightIndex].expected = round(prevLayer[weightIndex].expected);
        });
      });
    }
  }
};

run([0.2, 0.7]);
printNetwork();
debugger;

const setExpected = x => round((Math.random() + x) / 2);
for (let i = 0; i < 999; i++) {
  let expected = [Math.floor(Math.random() * 2)]; // config expected result
  expected[1] = expected[0] === 0 ? 1 : 0;
  const input = [setExpected(expected[0]), setExpected(expected[1])];
  weights = run(input, expected);
  // console.log(input, expected);
  // console.log(weights.join('\t'));
}

run([0.2, 0.7]); // check
printNetwork();
debugger;

// const propagated_error = weight * weightsDelta;
