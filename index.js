let weights = [0.42, 0.17, 0.78, 0.55, 0.22, 0.99, 0.12, 0.88]; // random weights
// const error = 0.05
const learn_rate = 0.1;
const round = x => Math.round(x * 10000) / 10000; // 0.42851.. >> 0.43
const sigmoid = x => round(1 / (1 + Math.exp(-x))); // range: 0 - 1

// const config = { input: 2, neurons: [2], output: 2 };
//  i1 - w1 - n1 - w5 - r1 (e1)
//     - w2 -    - w6 -
//  i2 - w3 - n2 - w7 - r2 (e2)
//     - w4 -    - w8 -

const adjustWeight = (actual, expected, old_weight, input) => {
  const error = Math.abs(actual - expected);
  const weights_delta = round(error * actual * (1 - actual)); // youtu.be/HA-F6cZPvrg?t=14m30s
  const new_weight = round(old_weight - input * weights_delta * learn_rate);
  if (actual < expected) {
    console.log(123, actual, expected, input, old_weight, '->', new_weight);
  }
  // console.log(error, `old:${old_weight} new:${new_weight}`, weights_delta);
  return new_weight;
};

const run = (train, weights, input, expected = []) => {
  const [i1, i2] = input;
  const [w1, w2, w3, w4, w5, w6, w7, w8] = weights;
  // const n1 = sigmoid(i1 * w1 + i2 * w3); // neurons
  // const n2 = sigmoid(i1 * w2 + i2 * w4);
  // const r1 = sigmoid(n1 * w5 + n2 * w7); // result
  // const r2 = sigmoid(n1 * w6 + n2 * w8);

  const n1 = i1 * (w1 + 0.5);

  let logString = `input:[${i1},${i2}] result:[${r1},${r2}]`;
  let logWeights = `weights:\t${weights.join('\t')}`;
  if (train) {
    const updatedWeights = weights.slice();
    const [e1, e2] = expected;
    logString += ` expected:[${e1},${e2}]`;
    updatedWeights[4] = adjustWeight(r1, e1, w5, n1);
    updatedWeights[5] = adjustWeight(r2, e2, w6, n1);
    updatedWeights[6] = adjustWeight(r1, e1, w7, n2);
    updatedWeights[7] = adjustWeight(r2, e2, w8, n2);
    const newN1 = sigmoid(r1 * w5 + r2 * w6); // ? use e1/e2 instead of r1/r2
    const newN2 = sigmoid(r1 * w7 + r2 * w8);
    updatedWeights[0] = adjustWeight(n1, newN1, w1, i1);
    updatedWeights[1] = adjustWeight(n2, newN2, w2, i1);
    updatedWeights[2] = adjustWeight(n1, newN1, w3, i2);
    updatedWeights[3] = adjustWeight(n2, newN2, w4, i2);
    logWeights += `\nupdatedWeights:\t${updatedWeights.join('\t')}`;
    // setTimeout(() => run(0, updatedWeights, input), 100);
    return updatedWeights;
  }
  console.log(`${logString}\n${logWeights}`);
};

const setExpected = x => round((Math.random() + x) / 2);
run(0, weights, [0.2, 0.7]);
for (let i = 0; i < 3; i++) {
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

// const propagated_error = weight * weights_delta;
