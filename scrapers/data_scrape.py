{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Downloading data...\n",
      "Success! Exported 5690 companies with 31 columns.\n"
     ]
    }
   ],
   "source": [
    "import pandas as pd\n",
    "import requests\n",
    "\n",
    "# 1. Fetch the data\n",
    "url = \"https://yc-oss.github.io/api/companies/all.json\"\n",
    "print(\"Downloading data...\")\n",
    "response = requests.get(url)\n",
    "data = response.json()\n",
    "\n",
    "# 2. Load into Pandas\n",
    "# This automatically handles the JSON structure\n",
    "df = pd.DataFrame(data)\n",
    "\n",
    "# 3. Basic Cleaning for your Capstone\n",
    "# Convert 'batch' (e.g., S24) into a more readable 'Season' and 'Year'\n",
    "df['season'] = df['batch'].str[0] # 'S' or 'W'\n",
    "df['year'] = \"20\" + df['batch'].str[1:] # '2024'\n",
    "\n",
    "# 4. Save to CSV\n",
    "df.to_csv(\"yc_companies_complete.csv\", index=False)\n",
    "print(f\"Success! Exported {len(df)} companies with {len(df.columns)} columns.\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python (learn-env)",
   "language": "python",
   "name": "learn-env"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.8"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
