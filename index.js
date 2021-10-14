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

const network = [];
const init = (l1, l2) => {
  // console.log('init', l1, l2);
  const layerWeights = [];
  for (let i = 0; i < l1; i++) {
    layerWeights[i] = [];
    for (let j = 0; j < l2; j++) {
      layerWeights[i].push(round(Math.random()));
    }
  }
  network.push(layerWeights);
};
const config = { layers: [2, 3, 2] };
const { layers } = config;
layers.forEach((i, index) => layers[index + 1] && init(layers[index], layers[index + 1]));

console.log('NETWORK:', network.map(layer => JSON.stringify(layer)).join('\n'));

const run = (train, weights_old, initialInput, expected = []) => {
  const values = [initialInput]; // intermediate projected values - set initial input
  network.forEach((layer, layerIndex) => {
    const input = values[layerIndex];
    const valuesNext = [];
    values.push(valuesNext); // set next values layer

    const projectedWeights = [];
    debugger;
    layer.forEach((neuron, neuronIndex) => {
      projectedWeights.push(neuron.map(weight => round(projectWeight(input[neuronIndex], weight))));
    });
    debugger;

    // for (let i = 0; i<projectedWeights.length; i+=)
    projectedWeights.forEach(inputResults => {
      inputResults.forEach((result, index) => {
        if (!valuesNext[index]) {
          valuesNext[index] = result;
        } else {
          valuesNext[index] += result;
        }
        // if (index === 0) {
        //   console.log(111, result, valuesNext[index]);
        // }
      });
    });
    //
    console.log(`projectedWeights ${JSON.stringify(projectedWeights)}`);
    // console.log(values[layerIndex].length, layer.length);
    // console.log(`values1 ${JSON.stringify(values)}`);
    valuesNext = valuesNext.map(i => {
      return round(i / (values[layerIndex].length * layer.length));
    });
    // console.log(`values2 ${JSON.stringify(values)}`);
  });
  console.log(`values:${JSON.stringify(values)}`);
  // process.exit();

  const [i1, i2] = input;
  const [w1, w2, w3, w4, w5, w6, w7, w8] = weights;

  const n1 = round((projectWeight(i1, w1) + projectWeight(i2, w3)) / 2);
  const n2 = round((projectWeight(i1, w2) + projectWeight(i2, w4)) / 2);

  const r1 = round((projectWeight(n1, w5) + projectWeight(n2, w7)) / 2);
  const r2 = round((projectWeight(n1, w6) + projectWeight(n2, w8)) / 2);

  let logString = `input:[${i1},${i2}] result:[${r1},${r2}]`;
  let logWeights = `weights:\t${weights.join('\t')}`;
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
    // setTimeout(() => run(0, updatedWeights, input), 100);
    return updatedWeights;
  }
  console.log(`${logString}\n${logWeights}`);
};

const setExpected = x => round((Math.random() + x) / 2);
run(0, weights, [0.2, 0.7]);
for (let i = 0; i < 333333; i++) {
  // for (let i = 0; i < 9999; i++) {
  let expected = [Math.floor(Math.random() * 2)]; // config expected result
  expected[1] = expected[0] === 0 ? 1 : 0;
  const input = [setExpected(expected[0]), setExpected(expected[1])];
  weights = run(1, weights, input, expected);
  // console.log(input, expected);
  // console.log(weights.join('\t'));
}
run(0, weights, [0.2, 0.7]); // check

// adjustWeight(0.69, 0, 0.5, 0.77);
// const propagated_error = weight * weightsDelta;
