# Perceived vs. Measured Saftey Across Countries: A Report Generating Agent

**N.B: This is an early-stage, partially-completed project.**

This repopsitory contains Python package implementing a custom AI agent designed to produce 1-2 page reports on perceived versus measured safety in different countries. It can take as input user prompts that ask for this information in different ways. It grounds its answers in three different data sources:

1. Wikipedia country summaries, accessed via the wikipedia API.
2. World Development Index (WDI) data from 266 countries that includes 22 different mortality-related fields, collected in the years 2019, 2021, and 2023.
3. Individual survey responses from the Gallup World Risk Poll from 2019, 2021, and 2023.

The agent has access to a suite of tools that it can use to produce its report:

1. A set of tools that allow it to query which countries and fields exist in all of the different data sets.
2. A tool that allows it to view the Wikipedia summary for a given country.
3. A tool that allows it to view WDI mortality data for a given country and year.
4. A set of tools that allows it to view 50 randomly-selected survey responses from the world risk poll for a particular set of countries, years, and set of fields contained within those data sets.
5. A tool that allows it to obtain a quantitative measure of the extent to which people worry about harms versus the extent to which they actually experience them in a given country and year, as obtained via the world risk poll.

Data is downloaded from a HuggingFace repository when the package is loaded. 
