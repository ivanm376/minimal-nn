let weights = [0.42, 0.17, 0.78, 0.55, 0.22, 0.99, 0.12, 0.88]; // random weights
// const error = 0.05
const learnRate = 0.02;
const round = x => Math.round(x * 1000) / 1000; // 0.42851.. >> 0.43
const sigmoid = x => round(1 / (1 + Math.exp(-x))); // range: 0 - 1

// const config = { input: 2, neurons: [2], output: 2 };
//  i1 - w1 - n1 - w5 - r1 (e1)
//     - w2 -    - w6 -
//  i2 - w3 - n2 - w7 - r2 (e2)
//     - w4 -    - w8 -
// i     w     r
// 0.6   1   > 1
// 0.6   0.5 > 0.6
// 0.6   0   > 0

// 0.0   0   > 1
// 0.1   x   > 0.9

const projectWeight_old = (input, weight) => (input * (weight + 0.5) * 2) / 3;
const calculateWeight = (input, result) => (result * 3) / input / 2 - 0.5;
const calculateInput = (result, weight) => (result * 3) / 2 / (weight + 0.5);
const adjustWeight_old = (input, oldWeight, result, expected) => {
  const error = round((expected - result) * learnRate);
  const newExpectedResult = round(result + error);

  // const weightsDelta = round(error * result * (1 - result)); // youtu.be/HA-F6cZPvrg?t=14m30s
  // const newWeight = round(oldWeight - input * weightsDelta * learnRate);
  const oldProjectedResult = round(projectWeight(input, oldWeight));
  const newProjectedResult = round((newExpectedResult / result) * oldProjectedResult);
  const calculatedWeight = round(calculateWeight(input, newProjectedResult));
  console.log(
    `${expected} :\t${error}\t${result}\t-> ${newExpectedResult}\t${oldProjectedResult}\t-> ${newProjectedResult}\tinput:${input}\tw:${calculatedWeight}`
  );
  return calculatedWeight;
  if (result > expected) {
    // console.log(weightsDelta, `${input}\t${result}\t${expected}\t${oldWeight}\t-> ${newWeight}`);
  }
  // console.log(error, `old:${oldWeight} new:${newWeight}`, weightsDelta);
  // return round(newWeight);

  // TODO reduce remaining part of weight eg 0.6 + (-1 * 0.4*learnRate)
};

const adjustWeight = (oldWeight, increase) => {
  if (increase) {
    return round(oldWeight + (1 - oldWeight) * learnRate);
  } else {
    return round(oldWeight - oldWeight * learnRate);
  }
};

const run = (train, weights, input, expected = []) => {
  const [i1, i2] = input;
  const [w1, w2, w3, w4, w5, w6, w7, w8] = weights;
  // const n1 = sigmoid(i1 * w1 + i2 * w3); // neurons
  // const n2 = sigmoid(i1 * w2 + i2 * w4);
  // const r1 = sigmoid(n1 * w5 + n2 * w7); // result
  // const r2 = sigmoid(n1 * w6 + n2 * w8);

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
    // updatedWeights[4] = adjustWeight(n1, w5, r1, e1);
    // updatedWeights[5] = adjustWeight(n1, w6, r2, e2);
    // updatedWeights[6] = adjustWeight(n2, w7, r1, e1);
    // updatedWeights[7] = adjustWeight(n2, w8, r2, e2);

    updatedWeights[4] = adjustWeight(w5, r1 < e1);
    updatedWeights[5] = adjustWeight(w6, r2 < e2);
    updatedWeights[6] = adjustWeight(w7, r1 < e1);
    updatedWeights[7] = adjustWeight(w8, r2 < e2);

    // const newN1 = sigmoid(r1 * w5 + r2 * w6); // ? use e1/e2 instead of r1/r2
    // const newN2 = sigmoid(r1 * w7 + r2 * w8);

    const newN1 = round((calculateInput(e1, updatedWeights[4]) + calculateInput(e2, updatedWeights[5])) / 2);
    console.log(e1, updatedWeights[4], e2, updatedWeights[5], ':', n1, newN1);
    const newN2 = round((calculateInput(e1, updatedWeights[6]) + calculateInput(e2, updatedWeights[7])) / 2);

    updatedWeights[0] = adjustWeight(w1, n1 < newN1);
    updatedWeights[1] = adjustWeight(w2, n2 < newN2);
    updatedWeights[2] = adjustWeight(w3, n1 < newN1);
    updatedWeights[3] = adjustWeight(w4, n2 < newN2);

    // const newN1 = sigmoid(r1 * w5 + r2 * w6); // ? use e1/e2 instead of r1/r2
    // const newN2 = sigmoid(r1 * w7 + r2 * w8);
    // updatedWeights[0] = adjustWeight(n1, newN1, w1, i1);
    // updatedWeights[1] = adjustWeight(n2, newN2, w2, i1);
    // updatedWeights[2] = adjustWeight(n1, newN1, w3, i2);
    // updatedWeights[3] = adjustWeight(n2, newN2, w4, i2);
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

// const propagated_error = weight * weightsDelta;
