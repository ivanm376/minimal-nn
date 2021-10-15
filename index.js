let weights = [0.42, 0.17, 0.78, 0.55, 0.22, 0.99, 0.12, 0.88]; // random weights
// const error = 0.05
const learnRate = 0.02;
const round = x => Math.round(x * 1000) / 1000; // 0.42851.. >> 0.43
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
const calculateInput = (result, weight) => {
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

const config = { layers: [2, 3, 2] };
const { layers } = config;
let weightId = 0; // weights count
let neuronId = 0;
const network = [];
layers.forEach((neuronsCountCurrentLayer, index) => {
  const neuronsCountPrevLayer = layers[index - 1];
  const neuronsCountNextLayer = layers[index + 1];
  const layer = [];
  for (let i = 0; i < neuronsCountCurrentLayer; i++) {
    const neuron = { input: [], output: [], neuronId: neuronId++, layerId: index };
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
    id: neuron.neuronId,
    layer: `L${neuron.layerId}`,
    input: neuron.input.map(weight => `id:${weight.id} value:${weight.value}`).join(' | '),
    value: neuron.value || 0,
    expected: neuron.expected || 0, // expected value
    output: neuron.output.map(weight => `id:${weight.id} value:${weight.value}`).join('\t| '),
  });
  console.table(network.reduce((a, b) => a.concat(b)).map(layerMap));
};
// printNetwork();debugger;

const run = (input, expected) => {
  network[0].forEach((neuron, index) => (neuron.value = input[index])); // set initial input

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
  if (expected) {
    printNetwork();
    debugger;
    network[network.length - 1].forEach((neuron, index) => (neuron.expected = expected[index]));
    printNetwork();
    debugger;
    for (let layerIndex = network.length - 1; layerIndex >= 0; layerIndex--) {
      console.log(layerIndex);
      const layer = network[layerIndex];
      const prevLayer = network[layerIndex - 1] || [];
      console.log(layerIndex, layer, prevLayer);
      layer.forEach((neuron, neuronIndex) => {
        neuron.output.forEach((weight, weightIndex) => {
          // console.log(round(projectWeight(neuron.value, weight.value)));
          prevLayer[weightIndex].value += projectWeight(neuron.value, weight.value);
          nextLayer[weightIndex].value = round(nextLayer[weightIndex].value);
        });
      });

      adjustWeight(weight, result < expectedResult[resultIndex]);
    }
    printNetwork();
    debugger;
    ///

    const valuesReversed = values.slice().reverse();
    const networkReversed = JSON.parse(JSON.stringify(network)).slice().reverse();
    const networkReversedCopy = JSON.parse(JSON.stringify(networkReversed));
    let expectedResult = expected;
    valuesReversed.forEach((results, reversedIndex) => {
      results.forEach((result, resultIndex) => {
        networkReversedCopy[reversedIndex][resultIndex] = networkReversed[reversedIndex][resultIndex].map(weight =>
          adjustWeight(weight, result < expectedResult[resultIndex])
        );
        debugger;
      });
    });
  }
  debugger;

  // const [i1, i2] = input;
  // const [w1, w2, w3, w4, w5, w6, w7, w8] = weights;
  // const n1 = round((projectWeight(i1, w1) + projectWeight(i2, w3)) / 2);
  // const n2 = round((projectWeight(i1, w2) + projectWeight(i2, w4)) / 2);
  // const r1 = round((projectWeight(n1, w5) + projectWeight(n2, w7)) / 2);
  // const r2 = round((projectWeight(n1, w6) + projectWeight(n2, w8)) / 2);
  // let logString = `input:[${i1},${i2}] result:[${r1},${r2}]`;
  // let logWeights = `weights:\t${weights.join('\t')}`;
  if (train) {
    const updatedWeights = weights.slice();
    const [e1, e2] = expected;
    logString += ` expected:[${e1},${e2}]`;

    updatedWeights[4] = adjustWeight(w5, r1 < e1);
    updatedWeights[5] = adjustWeight(w6, r2 < e2);
    updatedWeights[6] = adjustWeight(w7, r1 < e1);
    updatedWeights[7] = adjustWeight(w8, r2 < e2);

    const newN1 = round((calculateInput(e1, updatedWeights[4]) + calculateInput(e2, updatedWeights[5])) / 2);
    // console.log(e1, updatedWeights[4], e2, updatedWeights[5], ':', n1, newN1);
    const newN2 = round((calculateInput(e1, updatedWeights[6]) + calculateInput(e2, updatedWeights[7])) / 2);

    updatedWeights[0] = adjustWeight(w1, n1 < newN1);
    updatedWeights[1] = adjustWeight(w2, n2 < newN2);
    updatedWeights[2] = adjustWeight(w3, n1 < newN1);
    updatedWeights[3] = adjustWeight(w4, n2 < newN2);

    logWeights += `\nupdatedWeights:\t${updatedWeights.join('\t')}`;
    // setTimeout(() => run(input), 100);
    return updatedWeights;
  }
  console.log(`${logString}\n${logWeights}`);
};

const setExpected = x => round((Math.random() + x) / 2);
// run([0.2, 0.7]);
for (let i = 0; i < 3; i++) {
  let expected = [Math.floor(Math.random() * 2)]; // config expected result
  expected[1] = expected[0] === 0 ? 1 : 0;
  const input = [setExpected(expected[0]), setExpected(expected[1])];
  weights = run(input, expected);
  // console.log(input, expected);
  // console.log(weights.join('\t'));
}
run([0.2, 0.7]); // check

// adjustWeight(0.69, 0, 0.5, 0.77);
// const propagated_error = weight * weightsDelta;
