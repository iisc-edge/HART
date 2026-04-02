from whar_datasets import (
    Loader,
    LOSOSplitter,
    PostProcessingPipeline,
    PreProcessingPipeline,
    TorchAdapter,
    WHARDatasetID,
    get_dataset_cfg,
)

for dataset_id in WHARDatasetID:

    print(f"\nProcessing: {dataset_id.name}")
    
    cfg = get_dataset_cfg(dataset_id)
    pre_pipeline = PreProcessingPipeline(cfg)
    activity_df, session_df, window_df = pre_pipeline.run()

    print("Preprocessing Done.")
