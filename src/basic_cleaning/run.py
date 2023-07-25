#!/usr/bin/env python
"""
Download from W&B the raw dataset and apply some basic data cleaning,
exporting the result to a new artifact
"""
import argparse
import logging
import pandas as pd
import wandb


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    logger.info("Loading Artifact from WandB")
    local_path = run.use_artifact(args.input_artifact).file()

    df = pd.read_csv(local_path)

    logger.info(f"Cleaning Dataset {args.input_artifact}")

    min_price = args.min_price
    max_price = args.max_price

    # Drop Outlier
    idx = df['price'].between(min_price, max_price)
    df = df[idx].copy()

    # Convert format of date from string to datetime
    df['last_review'] = pd.to_datetime(df['last_review'])

    idx = (df['longitude'].between(-74.25, -73.50)
           & df['latitude'].between(40.5, 41.2))
    df = df[idx].copy()

    # Save cleaned dataframe
    df.to_csv("clean_sample.csv", index=False)

    logger.info("Store dataset in artifact.")
    artifact = wandb.Artifact(
            args.output_artifact,
            type=args.output_type,
            description=args.output_description
            )

    artifact.add_file("clean_sample.csv")

    run.log_artifact(artifact)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="A very basic data cleaning")

    parser.add_argument(
        "--input_artifact",
        type=str,
        help="Data that will be cleaned",
        required=True
    )

    parser.add_argument(
        "--output_artifact",
        type=str,
        help="Name of artifact where the cleaned data will be store",
        required=True
    )

    parser.add_argument(
        "--output_type",
        type=str,
        help="Output type of artifact",
        required=True
    )

    parser.add_argument(
        "--output_description",
        type=str,
        help="Description of cleaned data",
        required=True
    )

    parser.add_argument(
        "--min_price",
        type=float,
        help="Minimum price",
        required=True
    )

    parser.add_argument(
        "--max_price",
        type=float,
        help="Maximum price",
        required=True
    )

    args = parser.parse_args()

    go(args)
