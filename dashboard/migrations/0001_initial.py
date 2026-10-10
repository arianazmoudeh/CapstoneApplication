
import django.db.models.deletion
from django.conf import settings
from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.CreateModel(
            name='PriceSnapshot',
            fields=[
                ('key', models.CharField(max_length=30, primary_key=True, serialize=False)),
                ('data', models.JSONField(default=dict)),
                ('fetched_at', models.DateTimeField()),
                ('refresh_after', models.DateTimeField()),
            ],
        ),
        migrations.CreateModel(
            name='Portfolio',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(default='My crypto portfolio', max_length=80)),
                ('budget', models.DecimalField(decimal_places=2, max_digits=14)),
                ('weights', models.JSONField(default=dict)),
                ('updated_at', models.DateTimeField(auto_now=True)),
                ('user', models.OneToOneField(on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL)),
            ],
        ),
    ]
