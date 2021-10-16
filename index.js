// const error = 0.05
const learnRate = 0.002; // 0.2 %
const round = x => Math.round(x * 10000) / 10000; // 0.42851.. >> 0.43

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
const joinString = i => i.join(' | ').slice(0, 60);
const printNetwork = () => {
  const layerMap = neuron => ({
    id: neuron.id,
    layer: `L${neuron.layerId}`,
    input: joinString(neuron.input.map(weight => `id:${weight.id} value:${weight.value}`)),
    value: neuron.value,
    expected: neuron.expected, // expected value
    output: joinString(neuron.output.map(weight => `id:${weight.id} value:${weight.value}`)),
  });
  console.table(network.reduce((a, b) => a.concat(b)).map(layerMap));
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
        const addedValue = projectWeight(neuron.value, weight.value);
        // console.log(
        //   `${neuron.id} > ${nextLayer[weightIndex].id} : projectWeight(${neuron.value}`,
        //   `, ${weight.value})=${round(addedValue)}`
        // );
        ///// nextLayer[weightIndex] === nextLayer.find(i => i.input.find(j => j.id === weight.id))
        nextLayer[weightIndex].value = round(nextLayer[weightIndex].value + addedValue);
      });
    });
    nextLayer.forEach(neuron => (neuron.value = round(neuron.value / layer.length)));
  });

  let error = [];

  // TRAIN:
  if (expected.length) {
    // printNetwork();
    // debugger;
    for (let layerIndex = network.length - 1; layerIndex >= 0; layerIndex--) {
      const layer = network[layerIndex];
      const prevLayer = network[layerIndex - 1] || [];
      layer.forEach((neuron, neuronIndex) => {
        if (layerIndex === network.length - 1) {
          // debugger;
          if (neuron.expected === 1) {
            error.unshift(`\t${neuron.expected}\t${neuron.value}\t`);
          } else {
            error.push(`\t${neuron.expected}\t${neuron.value}\t`);
          }
        }
        neuron.input.forEach((weight, weightIndex) => {
          const addedValue = projectWeight(prevLayer[weightIndex].value, weight.value);
          const delta = neuron.expected - addedValue;
          //// const delta = neuron.value - addedValue;
          const oldWValue = weight.value;
          let newWValue = weight.value;
          if (delta > 0) {
            newWValue += (1 - weight.value) * (delta / neuron.value) * learnRate; // increase
          } else {
            newWValue += weight.value * (delta / neuron.value) * learnRate; // decrease
          }
          if (newWValue < 0 || newWValue > 1) {
            debugger;
          }
          // const b = projectWeight(prevLayer[weightIndex].value, newWValue);

          weight.value = round(newWValue);

          // console.log(
          //   `${prevLayer[weightIndex].id} > ${neuron.id} : projectWeight(${prevLayer[weightIndex].value}`,
          //   `, ${weight.value})=${round(addedValue)}`
          // );
          // debugger;

          const addedExpected = projectWeightBack(addedValue, newWValue);
          // console.log(round(prevLayer[weightIndex].expected), '+', round(addedExpected));
          prevLayer[weightIndex].expected = round(prevLayer[weightIndex].expected + addedExpected);
          // printNetwork();
          // debugger;
        });
      });
      prevLayer.forEach(neuron => (neuron.expected = round(neuron.expected / layer.length)));
      // printNetwork();
      // debugger;
    }
  }
  // printNetwork();
  // debugger;
  return error;
};

run([0.2, 0.7]);
printNetwork();

const setExpected = x => round((Math.random() + x) / 2);
for (let i = 0; i < 99999; i++) {
  const expected = [Math.floor(Math.random() * 2)]; // config expected result
  expected[1] = expected[0] === 0 ? 1 : 0;
  const error = run([setExpected(expected[0]), setExpected(expected[1])], expected);

  if (i % 10 === 0) {
    // console.log(i, '\t', error.join('\t'));
  }
}

run([0.2, 0.7]);
printNetwork();
debugger;

// const propagated_error = weight * weightsDelta;
