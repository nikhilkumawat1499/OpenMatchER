# UI Test Datasets

Use these files from the OpenMatchER UI upload control:

- `sample_companies.csv`: select `name` as the matching column.
- `sample_people.csv`: select `name` as the matching column.
- `sample_products.csv`: select `name` as the matching column.
- `sample_organizations.json`: select `name` as the matching column.

The files are intentionally small and include near-duplicates so the cluster review panel has matches to show after you run a pipeline.

Recommended UI flow:

1. Open `http://localhost:3000`.
2. Create a project.
3. Upload one file from this `datasets/` folder.
4. Select the uploaded dataset.
5. Choose `name` as the match field.
6. Use a threshold around `0.75` to `0.86`.
7. Click `Run`.

