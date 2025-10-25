#!/usr/bin/env python3
"""
Script to download relevant agricultural datasets from Kaggle
for enhancing the AgriNathi ML system
"""

import os
import kaggle
import pandas as pd
from pathlib import Path

def setup_kaggle_credentials():
    """Setup Kaggle API credentials"""
    kaggle_dir = Path.home() / '.kaggle'
    kaggle_dir.mkdir(exist_ok=True)

    # Check if kaggle.json exists
    kaggle_json = kaggle_dir / 'kaggle.json'
    if not kaggle_json.exists():
        print("Kaggle API credentials not found!")
        print("Please:")
        print("1. Go to https://www.kaggle.com/account")
        print("2. Click 'Create New API Token'")
        print("3. Save the downloaded kaggle.json file to:", kaggle_dir)
        print("4. Run this script again")
        return False

    # Set proper permissions
    os.chmod(kaggle_json, 0o600)
    return True

def download_agricultural_datasets():
    """Download relevant agricultural datasets from Kaggle"""

    datasets = [
        # Plant diseases and pest datasets
        {
            'name': 'emmarex/plant-disease',
            'description': 'Plant disease recognition dataset'
        },
        {
            'name': 'vipoooool/new-plant-diseases-dataset',
            'description': 'New plant diseases dataset with 38 classes'
        },
        {
            'name': 'rishabkoul1/plant-disease-dataset',
            'description': 'Plant disease detection dataset'
        },
        {
            'name': 'samarthbordikar/agricultural-pests-dataset',
            'description': 'Agricultural pests image dataset'
        },
        {
            'name': 'nancyalaswad90/pest-dataset',
            'description': 'Pest identification dataset'
        },
        {
            'name': 'nancyalaswad90/disease-dataset',
            'description': 'Disease information dataset'
        },

        # Crop and farming data
        {
            'name': 'siddharthss/crop-recommendation-dataset',
            'description': 'Crop recommendation dataset with soil and climate data'
        },
        {
            'name': 'atharvaingle/crop-recommendation-dataset',
            'description': 'Crop recommendation data'
        },
        {
            'name': 'nancyalaswad90/fertilizer-prediction',
            'description': 'Fertilizer prediction dataset'
        },
        {
            'name': 'nancyalaswad90/crop-dataset',
            'description': 'Crop yield and management dataset'
        },
        {
            'name': 'nancyalaswad90/farming-dataset',
            'description': 'General farming practices dataset'
        },

        # Weather and climate data
        {
            'name': 'nancyalaswad90/weather-dataset',
            'description': 'Weather data for agricultural planning'
        },
        {
            'name': 'nancyalaswad90/climate-dataset',
            'description': 'Climate change impact on agriculture'
        },

        # Soil data
        {
            'name': 'nancyalaswad90/soil-dataset',
            'description': 'Soil analysis dataset'
        },
        {
            'name': 'nancyalaswad90/soil-fertility-dataset',
            'description': 'Soil fertility and nutrient management'
        },

        # Livestock and animal husbandry
        {
            'name': 'nancyalaswad90/livestock-dataset',
            'description': 'Livestock management and health dataset'
        },
        {
            'name': 'nancyalaswad90/animal-husbandry-dataset',
            'description': 'Animal husbandry practices dataset'
        },

        # Irrigation and water management
        {
            'name': 'nancyalaswad90/irrigation-dataset',
            'description': 'Irrigation systems and water management'
        },
        {
            'name': 'nancyalaswad90/water-management-dataset',
            'description': 'Sustainable water management for farming'
        },

        # Organic farming and sustainability
        {
            'name': 'nancyalaswad90/organic-farming-dataset',
            'description': 'Organic farming practices and certification'
        },
        {
            'name': 'nancyalaswad90/sustainable-agriculture-dataset',
            'description': 'Sustainable agriculture techniques'
        },

        # Market and economic data
        {
            'name': 'nancyalaswad90/market-prices-dataset',
            'description': 'Agricultural market prices and trends'
        },
        {
            'name': 'nancyalaswad90/farm-economics-dataset',
            'description': 'Farm economics and profitability analysis'
        },

        # Rural development and extension
        {
            'name': 'nancyalaswad90/rural-development-dataset',
            'description': 'Rural development and farmer support programs'
        },
        {
            'name': 'nancyalaswad90/extension-services-dataset',
            'description': 'Agricultural extension services data'
        },

        # Seed and planting materials
        {
            'name': 'nancyalaswad90/seed-dataset',
            'description': 'Seed varieties and planting materials'
        },
        {
            'name': 'nancyalaswad90/planting-materials-dataset',
            'description': 'Planting materials and propagation techniques'
        },

        # Post-harvest and storage
        {
            'name': 'nancyalaswad90/post-harvest-dataset',
            'description': 'Post-harvest handling and storage techniques'
        },
        {
            'name': 'nancyalaswad90/storage-dataset',
            'description': 'Crop storage and preservation methods'
        },

        # Farm machinery and technology
        {
            'name': 'nancyalaswad90/farm-machinery-dataset',
            'description': 'Farm machinery and equipment data'
        },
        {
            'name': 'nancyalaswad90/agritech-dataset',
            'description': 'Agricultural technology and innovation'
        },

        # Climate-smart agriculture
        {
            'name': 'nancyalaswad90/climate-smart-agriculture-dataset',
            'description': 'Climate-smart farming practices'
        },
        {
            'name': 'nancyalaswad90/adaptation-strategies-dataset',
            'description': 'Climate change adaptation strategies'
        },

        # Comprehensive crop and plant datasets for image recognition
        {
            'name': 'marquis03/plant-disease-recognition-dataset',
            'description': 'Comprehensive plant disease recognition with multiple crops'
        },
        {
            'name': 'abdallahalidev/plantvillage-dataset',
            'description': 'PlantVillage dataset with various crops and diseases'
        },
        {
            'name': 'prasunroy/natural-images',
            'description': 'Natural images dataset including plants and crops'
        },
        {
            'name': 'jorgebuenoperez/leaf-disease-dataset',
            'description': 'Leaf disease dataset for multiple plant species'
        },
        {
            'name': 'nancyalaswad90/crop-disease-dataset',
            'description': 'Crop disease identification dataset'
        },
        {
            'name': 'nancyalaswad90/fruit-disease-dataset',
            'description': 'Fruit disease recognition dataset'
        },
        {
            'name': 'nancyalaswad90/vegetable-disease-dataset',
            'description': 'Vegetable disease identification dataset'
        },

        # Specific crop datasets
        {
            'name': 'nancyalaswad90/maize-dataset',
            'description': 'Maize crop disease and health dataset'
        },
        {
            'name': 'nancyalaswad90/wheat-dataset',
            'description': 'Wheat crop disease and management dataset'
        },
        {
            'name': 'nancyalaswad90/rice-dataset',
            'description': 'Rice crop disease and cultivation dataset'
        },
        {
            'name': 'nancyalaswad90/potato-dataset',
            'description': 'Potato disease and production dataset'
        },
        {
            'name': 'nancyalaswad90/tomato-dataset',
            'description': 'Tomato disease and growth dataset'
        },
        {
            'name': 'nancyalaswad90/cotton-dataset',
            'description': 'Cotton crop disease and farming dataset'
        },
        {
            'name': 'nancyalaswad90/sugarcane-dataset',
            'description': 'Sugarcane disease and cultivation dataset'
        },

        # Fruit crop datasets
        {
            'name': 'nancyalaswad90/apple-dataset',
            'description': 'Apple fruit disease and quality dataset'
        },
        {
            'name': 'nancyalaswad90/banana-dataset',
            'description': 'Banana fruit disease and production dataset'
        },
        {
            'name': 'nancyalaswad90/orange-dataset',
            'description': 'Orange fruit disease and harvesting dataset'
        },
        {
            'name': 'nancyalaswad90/grape-dataset',
            'description': 'Grape fruit disease and vineyard dataset'
        },
        {
            'name': 'nancyalaswad90/mango-dataset',
            'description': 'Mango fruit disease and tropical farming dataset'
        },
        {
            'name': 'nancyalaswad90/pineapple-dataset',
            'description': 'Pineapple fruit disease and cultivation dataset'
        },

        # Vegetable crop datasets
        {
            'name': 'nancyalaswad90/cabbage-dataset',
            'description': 'Cabbage vegetable disease and farming dataset'
        },
        {
            'name': 'nancyalaswad90/carrot-dataset',
            'description': 'Carrot vegetable disease and root crop dataset'
        },
        {
            'name': 'nancyalaswad90/onion-dataset',
            'description': 'Onion vegetable disease and bulb crop dataset'
        },
        {
            'name': 'nancyalaswad90/lettuce-dataset',
            'description': 'Lettuce vegetable disease and leafy crop dataset'
        },
        {
            'name': 'nancyalaswad90/pepper-dataset',
            'description': 'Pepper vegetable disease and spice crop dataset'
        },

        # Legume and pulse datasets
        {
            'name': 'nancyalaswad90/bean-dataset',
            'description': 'Bean legume disease and nitrogen-fixing crop dataset'
        },
        {
            'name': 'nancyalaswad90/pea-dataset',
            'description': 'Pea legume disease and cool-season crop dataset'
        },
        {
            'name': 'nancyalaswad90/lentil-dataset',
            'description': 'Lentil pulse disease and drought-tolerant crop dataset'
        },
        {
            'name': 'nancyalaswad90/chickpea-dataset',
            'description': 'Chickpea legume disease and protein crop dataset'
        },

        # Oilseed and cash crop datasets
        {
            'name': 'nancyalaswad90/soybean-dataset',
            'description': 'Soybean oilseed disease and protein crop dataset'
        },
        {
            'name': 'nancyalaswad90/groundnut-dataset',
            'description': 'Groundnut oilseed disease and peanut crop dataset'
        },
        {
            'name': 'nancyalaswad90/sunflower-dataset',
            'description': 'Sunflower oilseed disease and ornamental crop dataset'
        },
        {
            'name': 'nancyalaswad90/coffee-dataset',
            'description': 'Coffee cash crop disease and beverage crop dataset'
        },
        {
            'name': 'nancyalaswad90/tea-dataset',
            'description': 'Tea cash crop disease and plantation crop dataset'
        },

        # Root and tuber crop datasets
        {
            'name': 'nancyalaswad90/cassava-dataset',
            'description': 'Cassava root crop disease and staple food dataset'
        },
        {
            'name': 'nancyalaswad90/sweet-potato-dataset',
            'description': 'Sweet potato root crop disease and nutritious tuber dataset'
        },
        {
            'name': 'nancyalaswad90/yam-dataset',
            'description': 'Yam tuber crop disease and African staple dataset'
        },

        # Spice and condiment crop datasets
        {
            'name': 'nancyalaswad90/chili-dataset',
            'description': 'Chili spice crop disease and condiment dataset'
        },
        {
            'name': 'nancyalaswad90/garlic-dataset',
            'description': 'Garlic spice crop disease and medicinal plant dataset'
        },
        {
            'name': 'nancyalaswad90/ginger-dataset',
            'description': 'Ginger spice crop disease and rhizome crop dataset'
        },
        {
            'name': 'nancyalaswad90/turmeric-dataset',
            'description': 'Turmeric spice crop disease and medicinal rhizome dataset'
        },

        # Forage and fodder crop datasets
        {
            'name': 'nancyalaswad90/alfalfa-dataset',
            'description': 'Alfalfa forage crop disease and livestock feed dataset'
        },
        {
            'name': 'nancyalaswad90/clover-dataset',
            'description': 'Clover forage crop disease and nitrogen-fixing forage dataset'
        },
        {
            'name': 'nancyalaswad90/napier-grass-dataset',
            'description': 'Napier grass forage crop disease and tropical fodder dataset'
        },

        # Medicinal and aromatic plant datasets
        {
            'name': 'nancyalaswad90/aloe-vera-dataset',
            'description': 'Aloe vera medicinal plant disease and therapeutic crop dataset'
        },
        {
            'name': 'nancyalaswad90/mint-dataset',
            'description': 'Mint aromatic plant disease and culinary herb dataset'
        },
        {
            'name': 'nancyalaswad90/basil-dataset',
            'description': 'Basil aromatic plant disease and culinary herb dataset'
        },

        # Plantation and perennial crop datasets
        {
            'name': 'nancyalaswad90/coconut-dataset',
            'description': 'Coconut plantation crop disease and multi-use palm dataset'
        },
        {
            'name': 'nancyalaswad90/rubber-dataset',
            'description': 'Rubber plantation crop disease and industrial crop dataset'
        },
        {
            'name': 'nancyalaswad90/palm-oil-dataset',
            'description': 'Palm oil plantation crop disease and biodiesel crop dataset'
        },

        # Indigenous and traditional crop datasets
        {
            'name': 'nancyalaswad90/moringa-dataset',
            'description': 'Moringa indigenous crop disease and superfood tree dataset'
        },
        {
            'name': 'nancyalaswad90/baobab-dataset',
            'description': 'Baobab indigenous crop disease and African superfruit dataset'
        },
        {
            'name': 'nancyalaswad90/sorghum-dataset',
            'description': 'Sorghum indigenous crop disease and drought-resistant cereal dataset'
        },
        {
            'name': 'nancyalaswad90/millet-dataset',
            'description': 'Millet indigenous crop disease and climate-resilient cereal dataset'
        }
    ]

    data_dir = Path('data/kaggle_datasets')
    data_dir.mkdir(parents=True, exist_ok=True)

    print("Starting download of agricultural datasets from Kaggle...")
    print(f"Datasets will be saved to: {data_dir.absolute()}")

    downloaded_datasets = []

    for dataset in datasets:
        try:
            print(f"\nDownloading: {dataset['name']}")
            print(f"Description: {dataset['description']}")

            # Create subdirectory for this dataset
            dataset_dir = data_dir / dataset['name'].split('/')[-1]
            dataset_dir.mkdir(exist_ok=True)

            # Download the dataset - try as competition first, then as dataset
            try:
                kaggle.api.competition_download_files(dataset['name'], path=str(dataset_dir), quiet=False)
            except:
                # If competition download fails, try as regular dataset
                kaggle.api.dataset_download_files(dataset['name'], path=str(dataset_dir), quiet=False, unzip=True)

            print(f"[OK] Successfully downloaded: {dataset['name']}")
            downloaded_datasets.append(dataset)

        except Exception as e:
            print(f"[ERROR] Failed to download {dataset['name']}: {str(e)}")
            continue

    print(f"\nDownload complete! Downloaded {len(downloaded_datasets)} out of {len(datasets)} datasets.")

    # Create a summary file
    summary_file = data_dir / 'downloaded_datasets.txt'
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("Downloaded Agricultural Datasets from Kaggle\n")
        f.write("=" * 50 + "\n\n")
        for dataset in downloaded_datasets:
            f.write(f"Dataset: {dataset['name']}\n")
            f.write(f"Description: {dataset['description']}\n")
            f.write(f"Location: data/kaggle_datasets/{dataset['name'].split('/')[-1]}\n")
            f.write("-" * 50 + "\n")

    print(f"Summary saved to: {summary_file}")

    return downloaded_datasets

def explore_downloaded_data():
    """Explore and analyze downloaded datasets"""
    data_dir = Path('data/kaggle_datasets')

    if not data_dir.exists():
        print("No downloaded datasets found. Run download_agricultural_datasets() first.")
        return

    print("\nExploring downloaded datasets...")

    for dataset_dir in data_dir.iterdir():
        if dataset_dir.is_dir() and dataset_dir.name != '__pycache__':
            print(f"\nDataset: {dataset_dir.name}")

            # Look for CSV files
            csv_files = list(dataset_dir.glob('*.csv'))
            if csv_files:
                print(f"  CSV files found: {len(csv_files)}")
                for csv_file in csv_files[:3]:  # Show first 3
                    try:
                        df = pd.read_csv(csv_file)
                        print(f"    {csv_file.name}: {len(df)} rows, {len(df.columns)} columns")
                        print(f"      Columns: {list(df.columns)[:5]}...")  # Show first 5 columns
                    except Exception as e:
                        print(f"    {csv_file.name}: Error reading - {str(e)}")

            # Look for image directories
            image_dirs = [d for d in dataset_dir.iterdir() if d.is_dir() and any(ext in str(d).lower() for ext in ['train', 'test', 'valid', 'images'])]
            if image_dirs:
                print(f"  Image directories found: {len(image_dirs)}")
                for img_dir in image_dirs[:3]:
                    image_count = len(list(img_dir.rglob('*.jpg')) + list(img_dir.rglob('*.png')) + list(img_dir.rglob('*.jpeg')))
                    print(f"    {img_dir.name}: {image_count} images")

def main():
    """Main function"""
    print("AgriNathi - Kaggle Agricultural Dataset Downloader")
    print("=" * 55)

    # Setup Kaggle credentials
    if not setup_kaggle_credentials():
        return

    # Download datasets
    downloaded = download_agricultural_datasets()

    if downloaded:
        # Explore downloaded data
        explore_downloaded_data()

        print("\n" + "=" * 55)
        print("Next steps:")
        print("1. Review the downloaded datasets in data/kaggle_datasets/")
        print("2. Integrate relevant data into the AgricultureMLService")
        print("3. Update the ML model training with new datasets")
        print("4. Test the enhanced agricultural advice system")
    else:
        print("No datasets were downloaded. Please check your Kaggle API credentials.")

if __name__ == '__main__':
    main()