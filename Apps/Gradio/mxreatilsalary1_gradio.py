# gradio app for running ML models

import hydra
from omegaconf import DictConfig
from src.experiment_model import MxRetailSalary1
import pandas as pd
import gradio as gr


@hydra.main(version_base=None, config_path="../../src/conf", config_name="config")
def main(cfg: DictConfig) -> None:
    experiment = MxRetailSalary1(cfg)
    
    def prediction(state, income, employees):
        input = pd.DataFrame([[state, float(income), int(employees)]], columns= experiment.feature_list)
        result = experiment.predict(input)
        return f"El salario (sin comisiones) es {result}"
    
    iface = gr.Interface(
        fn=prediction,
        inputs=[
            gr.inputs.Textbox(label="Estado Mx", default= 'Morelos'),
            gr.inputs.Slider(minimum=4000, maximum=10000, step=200, label="Ventas diarias estimadas"),
            gr.inputs.Slider(minimum=4, maximum=8, step=1, label="Empleados en el local")
        ],
        outputs=gr.outputs.Textbox(label="Resultado"),
        title="Gradio Salary Retail Estimator",
        description="Predice el salario (sin comisiones) en el sector retail."
    )
    
    iface.launch()


if __name__ == "__main__":
    main()
