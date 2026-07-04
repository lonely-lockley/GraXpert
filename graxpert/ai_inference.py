import logging
import os

import numpy as np
import onnxruntime as ort

from graxpert.ai_model_handling import get_execution_providers_ordered


def is_coreml_model(ai_path):
    return ai_path is not None and ai_path.endswith(".mlpackage") and os.path.isdir(ai_path)


class CoreMLInferenceSession:
    def __init__(self, ai_path, force_one_by_one=False):
        import coremltools as ct

        self.ai_path = ai_path
        self.ct = ct
        self.compute_units = ct.ComputeUnit.ALL
        self._load_model()
        self.logged_batch_fallback = False
        self.force_one_by_one = force_one_by_one

    def _load_model(self):
        self.model = self.ct.models.MLModel(self.ai_path, compute_units=self.compute_units)
        spec = self.model.get_spec()
        self.input_names = [input_description.name for input_description in spec.description.input]
        self.output_names = [output_description.name for output_description in spec.description.output]

    def run(self, inputs, return_first_output=True):
        input_values = {}
        for input_name in self.input_names:
            if input_name not in inputs:
                raise KeyError(f"Missing Core ML input '{input_name}'")
            input_values[input_name] = np.asarray(inputs[input_name], dtype=np.float32)

        batch_size = next(iter(input_values.values())).shape[0]

        if self.force_one_by_one and batch_size > 1:
            predictions = self._run_one_by_one(input_values)
        else:
            try:
                predictions = self.model.predict(input_values)
            except RuntimeError as err:
                self.force_one_by_one = True
                self._load_model()
                predictions = self._run_one_by_one(input_values, err)
            else:
                if self._has_non_finite_output(predictions):
                    self.force_one_by_one = True
                    self._load_model()
                    predictions = self._run_one_by_one(input_values)

        outputs = [predictions[output_name] for output_name in self.output_names]
        if return_first_output:
            return outputs[0]
        return outputs

    def _has_non_finite_output(self, predictions):
        return any(not np.isfinite(predictions[output_name]).all() for output_name in self.output_names)

    def _run_one_by_one(self, input_values, original_error=None):
        batch_size = next(iter(input_values.values())).shape[0]
        if batch_size <= 1:
            raise original_error or RuntimeError("Core ML one-by-one fallback requires a batch dimension larger than 1")

        if not self.logged_batch_fallback:
            if original_error is None:
                logging.info("Using Core ML one-by-one prediction.")
            else:
                logging.warning("Core ML batch prediction failed. Falling back to one-by-one prediction.")
            self.logged_batch_fallback = True
        output_batches = {output_name: [] for output_name in self.output_names}
        for batch_idx in range(batch_size):
            batch_inputs = {input_name: value[batch_idx : batch_idx + 1] for input_name, value in input_values.items()}
            predictions = self.model.predict(batch_inputs)
            for output_name in self.output_names:
                output_batches[output_name].append(predictions[output_name])

        return {output_name: np.concatenate(outputs, axis=0) for output_name, outputs in output_batches.items()}


class ONNXInferenceSession:
    def __init__(self, ai_path, ai_gpu_acceleration=True):
        providers = get_execution_providers_ordered(ai_gpu_acceleration)
        self.session = ort.InferenceSession(ai_path, providers=providers)
        logging.info(f"Available inference providers : {providers}")
        logging.info(f"Used inference providers : {self.session.get_providers()}")

    def run(self, inputs, return_first_output=True):
        outputs = self.session.run(None, inputs)
        if return_first_output:
            return outputs[0]
        return outputs


def create_inference_session(ai_path, ai_gpu_acceleration=True, force_one_by_one=False):
    if is_coreml_model(ai_path):
        logging.info(f"Using Core ML model: {ai_path}")
        return CoreMLInferenceSession(ai_path, force_one_by_one)

    return ONNXInferenceSession(ai_path, ai_gpu_acceleration)
