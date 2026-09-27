from __future__ import annotations

from mm_rcna.models.vision import VisionToolRunner


def main():
    # 你自己的标签
    lesion_labels = [
        "opacity",
        "effusion",
        "pneumothorax",
        "edema",
        "atelectasis",
        "cardiomegaly",
        "device",
    ]

    lung_regions = [
        "left_upper",
        "right_upper",
        "left_lower",
        "right_lower",
    ]

    # 初始化视觉模块
    runner = VisionToolRunner(
        image_encoder_name="cxr_foundation",
        vision_backbone_name="torchxrayvision_densenet121",
        lesion_labels=lesion_labels,
        lung_regions=lung_regions,
        image_size=224,
        device="cpu",   # 如果你有GPU并且环境配好了，可以改成 "cuda"
        checkpoint_path=None,
    )

    # 这里换成你自己的图像路径
    image_paths = [
        "./artifacts/images/CXR1_f.png"
    ]

    # 运行
    out = runner.run(image_paths)

    # 打印结果
    print("\n===== Vision Test Output =====")
    print("feature_vector length:", len(out.feature_vector))
    print("feature_vector[:10]:", out.feature_vector[:10])

    print("\nlesion_scores:")
    for k, v in out.lesion_scores.items():
        print(f"  {k}: {v:.4f}")

    print("\nregion_scores:")
    for k, v in out.region_scores.items():
        print(f"  {k}: {v:.4f}")

    print("\nquality_flags:", out.quality_flags)


if __name__ == "__main__":
    main()