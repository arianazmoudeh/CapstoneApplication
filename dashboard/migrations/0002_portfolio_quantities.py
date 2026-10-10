from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [("dashboard", "0001_initial")]

    operations = [
        migrations.AddField(
            model_name="portfolio",
            name="quantities",
            field=models.JSONField(default=dict),
        ),
    ]
