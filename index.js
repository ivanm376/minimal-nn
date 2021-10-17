// const error = 0.05
const sigmoid = x => 1 / (1 + Math.exp(-x));
const learnRate = 0.3;
const momentum = 0.1;
const round = x => x; //Math.round(x * 100000) / 100000; // 0.42851.. >> 0.43

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
const adjustWeight = (oldWeight, increase) => {
  if (increase) {
    return round(oldWeight + (1 - oldWeight) * learnRate);
  } else {
    return round(oldWeight - oldWeight * learnRate);
  }
};

let net;
net = {
  biases: [null, [-0.4424299895763397], [2.8855154514312744, -2.8785271644592285]],
  weights: [null, [[-5.431631088256836, 5.519097805023193]], [[-6.339480876922607], [6.325170040130615]]],
  errors: [
    { '0': -0.00015359566896222532, '1': -0.0007743531605228782 },
    { '0': -0.016961565241217613 },
    { '0': -0.5144253969192505, '1': 0.4831986427307129 },
  ],
  deltas: [
    { '0': -0.000002596405693111592, '1': -0.0001575937494635582 },
    { '0': -0.004239736590534449 },
    { '0': -0.12849929928779602, '1': 0.12066326290369034 },
  ],
  changes: [
    null,
    [{ '0': -0.00002187704194511752, '1': -0.0009101866744458675 }],
    [{ '0': -0.01951434090733528 }, { '0': 0.018324334174394608 }],
  ],
};

const config = { layers: [2, 1, 2] };
const { layers } = config;
let weightId = 0; // weights count
let neuronId = 0;
let network = [];
layers.forEach((neuronsCountCurrentLayer, index) => {
  const neuronsCountPrevLayer = layers[index - 1];
  const neuronsCountNextLayer = layers[index + 1];
  const layer = [];
  for (let i = 0; i < neuronsCountCurrentLayer; i++) {
    const neuron = { input: [], output: [], delta: 0, id: neuronId++, layerId: index };
    if (net) {
      neuron.bias = (net.biases[index] && net.biases[index][i]) || 0;
      neuron.delta = (net.deltas[index] && net.deltas[index][i]) || 0;
      neuron.error = (net.errors[index] && net.errors[index][i]) || 0;
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
        const weight = { id: weightId++ };
        const next = index + 1;
        weight.value = net && net.weights[next] && net.weights[next][j] && net.weights[next][j][i];
        weight.change = (net && net.changes[next] && net.changes[next][j][i]) || 0;
        if (!weight.value) {
          weight.value = Math.random() * 0.4 - 0.2; // brain.js:1413
        }
        neuron.output.push(weight);
      }
    }
    layer.push(neuron);
  }
  network.push(layer);
});
const joinString = i => i.join('\n'); //.slice(0, 70);
const weightString = w => `id:${w.id} value:${round(w.value)} change:${w.change}`;
const printNetwork = () => {
  const layerMap = neuron => ({
    id: neuron.id,
    layer: `L${neuron.layerId}`,
    input: joinString(neuron.input.map(weightString)),
    value: round(neuron.value),
    bias: round(neuron.bias),
    delta: round(neuron.delta),
    error: round(neuron.delta),
    expected: neuron.expected || '-', // expected value
    output: joinString(neuron.output.map(weightString)),
  });
  console.log(JSON.stringify(network));
  // console.table(network.flat(1).map(layerMap));
};
// printNetwork();debugger;

const run = (input, expected = []) => {
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
        // const addedValue = projectWeight(neuron.value, weight.value);
        // console.log(
        //   `${neuron.id} > ${nextLayer[weightIndex].id} : projectWeight(${neuron.value}`,
        //   `, ${weight.value})=${round(addedValue)}`
        // );
        ///// nextLayer[weightIndex] === nextLayer.find(i => i.input.find(j => j.id === weight.id))
        // nextLayer[weightIndex].value = round(nextLayer[weightIndex].value + addedValue);
        nextLayer[weightIndex].value = nextLayer[weightIndex].value + weight.value * neuron.value;
      });
    });
    // nextLayer.forEach(neuron => (neuron.value = round(neuron.value / layer.length)));
    nextLayer.forEach((neuron, neuronIndex) => {
      // console.log(neuron.bias === net.biases[layerIndex + 1][neuronIndex]); // - true
      let vv = neuron.value;
      neuron.value = sigmoid(neuron.bias + neuron.value); // bias + sum
    });
  });

  let errorLog = [];

  // TRAIN:
  if (expected.length) {
    // calculateDeltas:
    for (let layerIndex = network.length - 1; layerIndex >= 0; layerIndex--) {
      const layer = network[layerIndex];
      const nextLayer = network[layerIndex + 1] || [];
      layer.forEach((neuron, neuronIndex) => {
        // if (layerIndex === network.length - 1) {
        //   // debugger;
        //   if (neuron.expected === 1) {
        //     error.unshift(`\t${neuron.expected}\t${neuron.value}\t`);
        //   } else {
        //     error.push(`\t${neuron.expected}\t${neuron.value}\t`);
        //   }
        // }
        neuron.input.forEach((weight, weightIndex) => {
          // const addedValue = projectWeight(prevLayer[weightIndex].value, weight.value);
          let error = 0;
          // debugger;
          // neuron.value = 0.46613582968711853;
          if (layerIndex === network.length - 1) {
            error = neuron.expected - neuron.value;
            // 0 - 0.07301503180017563 = -0.07301503180017563
            // 1 - 0.9266279995532338 = 0.07337200044676617
            debugger;
            errorLog.push(error);
          } else {
            const deltas = nextLayer.map(i => i.delta); // this.deltas[layer + 1];
            // {0: -0.004941937505903706, 1: 0.004988455920769976} > 0
            debugger;
            for (let k = 0; k < deltas.length; k++) {
              error += deltas[k] * nextLayer[k].input[neuronIndex].value;
            }
          }
          // activeError[neuronIndex] = error; // neuron.error
          neuron.error = error;
          neuron.delta = error * neuron.value * (1 - neuron.value); // activeDeltas[neuronIndex]
          debugger;
          //////  0.004941937505903706  /  0.004988455920769976

          // //// const delta = neuron.value - addedValue;
          // const oldWValue = weight.value;
          // let newWValue = weight.value;
          // if (delta > 0) {
          //   newWValue += (1 - weight.value) * (delta / neuron.value) * learnRate; // increase
          // } else {
          //   newWValue += weight.value * (delta / neuron.value) * learnRate; // decrease
          // }
          // if (newWValue < 0 || newWValue > 1) {
          //   debugger;
          // }
          // // const b = projectWeight(prevLayer[weightIndex].value, newWValue);
          //
          // weight.value = round(newWValue);
          //
          // // console.log(
          // //   `${prevLayer[weightIndex].id} > ${neuron.id} : projectWeight(${prevLayer[weightIndex].value}`,
          // //   `, ${weight.value})=${round(addedValue)}`
          // // );
          // // debugger;
          //
          // const addedExpected = projectWeightBack(addedValue, newWValue);
          // // console.log(round(prevLayer[weightIndex].expected), '+', round(addedExpected));
          // prevLayer[weightIndex].expected = round(prevLayer[weightIndex].expected + addedExpected);
        });
      });
      // prevLayer.forEach(neuron => (neuron.expected = round(neuron.expected / layer.length)));
    }

    // adjustWeights:
    for (let layerIndex = 1; layerIndex < network.length; layerIndex++) {
      const layer = network[layerIndex];
      const prevLayer = network[layerIndex - 1] || [];
      const incoming = prevLayer.map(i => i.value);
      layer.forEach((neuron, neuronIndex) => {
        neuron.input.forEach((weight, weightIndex) => {
          let change = weight.change || 0; // activeChanges[node][k];
          weight.change = learnRate * neuron.delta * weight.value + momentum * change; // activeChanges[node][k]
          weight.value += change; // activeWeights[node][k]
        });
        neuron.bias += learnRate * neuron.delta;
      });
    }
  }
  return errorLog;
};

printNetwork();
debugger;

run([0.2, 0.6], [0, 1]);
printNetwork();
debugger;

const setExpected = x => round((Math.random() + x) / 2);
for (let i = 0; i < 4999; i++) {
  const expected = [Math.floor(Math.random() * 2)]; // config expected result
  expected[1] = expected[0] === 0 ? 1 : 0;
  const errorLog = run([setExpected(expected[0]), setExpected(expected[1])], expected);

  if (i % 100 === 0) {
    console.log(i, errorLog[0], errorLog[1]);
  }
}

run([0.2, 0.6]);
printNetwork();
debugger;

// const propagated_error = weight * weightsDelta;
