import gradio as gr

from space_demo import compare_policies_for_demo, run_live_episode


with gr.Blocks(title="MASS Startup Simulator") as demo:
    gr.Markdown(
        """
        # MASS: Multi-Agent Startup Simulator

        A long-horizon world-modeling environment where Tech, Growth, and Finance co-founders
        propose actions under partial observability, and a CEO chooses the final startup strategy.
        The trained CEO is a Qwen LoRA policy optimized from simulator rewards with GRPO.
        """
    )

    with gr.Tab("Live Episode"):
        with gr.Row():
            seed = gr.Number(value=7, precision=0, label="Seed")
            horizon = gr.Slider(5, 30, value=15, step=1, label="Horizon")
        run_button = gr.Button("Run Heuristic Multi-Agent Episode")
        episode_summary = gr.Textbox(label="Episode Summary", lines=3)
        episode_table = gr.Dataframe(
            headers=[
                "Day",
                "CEO Action",
                "Reward",
                "Money",
                "Users",
                "Quality",
                "Event",
                "Tech",
                "Growth",
                "Finance",
                "CEO Reasoning",
            ],
            label="Step Trace",
            wrap=True,
        )
        raw_json = gr.Code(label="Raw Episode JSON", language="json")
        run_button.click(
            run_live_episode,
            inputs=[seed, horizon],
            outputs=[episode_summary, episode_table, raw_json],
        )

    with gr.Tab("Training Result"):
        gr.Image("docs/assets/loss_curve.png", label="CEO Training Loss")
        gr.Image("docs/assets/reward_curve.png", label="Average Reward Comparison")
        gr.Image("docs/assets/reward_comparison.png", label="Before/After Metrics")
        compare_button = gr.Button("Show Baseline vs Trained CEO Metrics")
        comparison_summary = gr.Textbox(label="Interpretation", lines=4)
        comparison_table = gr.Dataframe(
            headers=["Metric", "Heuristic Baseline", "Trained CEO + Safety"],
            label="Policy Comparison",
        )
        compare_button.click(
            compare_policies_for_demo,
            inputs=[],
            outputs=[comparison_summary, comparison_table],
        )

    with gr.Tab("OpenEnv"):
        gr.Markdown(
            """
            The OpenEnv manifest is `openenv.yaml`.

            The environment exposes:

            - `reset()`
            - `step(action)`
            - `state`

            Valid actions:

            - `hire_employee`
            - `fire_employee`
            - `invest_in_product`
            - `run_marketing_campaign`
            - `do_nothing`
            - `pivot_strategy`
            """
        )


if __name__ == "__main__":
    demo.launch()
