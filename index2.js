// i     w     r
// 0.6   1   > 1
// 0.6   0.5 > 0.6
// 0.6   0   > 0

let input = 0.6;
let w1 = 0.6;
let result;

const projectWeight = (input, weight) => {
  // if (weight >= 1) {
  if (weight >= 0.5) {
    // weight === 1    > multiplier === 1
    // weight === 0.75 > multiplier === 0.5
    // weight === 0.5  > multiplier === 0
    return input + (1 - input) * (weight * 2 - 1);
  } else {
    // weight === 0.499 > multiplier === 0.9999 ~
    // weight === 0.25  > multiplier === 0.5
    // weight === 0     > multiplier === 0
    return input * weight * 2;
  }
};

const canvas = document.createElement('canvas');
document.body.appendChild(canvas);
const ctx = canvas.getContext('2d');
ctx.fillStyle = 'black';
const input = 0.3;
for (let x = 0; x <= 100; x += 1) {
  const y = projectWeight(input, x / 100);
  ctx.fillRect(x, Math.floor(y * 100), 1, 1); // dot
}
// <canvas id="tutorial" width="150" height="150"></canvas>;
