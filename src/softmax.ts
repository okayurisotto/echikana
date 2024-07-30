export const softmax = (values: readonly number[]): number[] => {
  const denominator = values
    .map((value) => Math.exp(value))
    .reduce((prev, current) => prev + current, 0);

  return values.map((value) => Math.exp(value) / denominator);
};
