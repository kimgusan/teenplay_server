# Generated by Django 5.0.2 on 2024-02-26 20:45

import django.db.models.deletion
import django.utils.timezone
from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        ('letter', '0002_rename_latter_letter_and_more'),
    ]

    operations = [
        migrations.CreateModel(
            name='ReceivedLetter',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('created_date', models.DateTimeField(auto_now_add=True)),
                ('updated_date', models.DateTimeField(default=django.utils.timezone.now)),
                ('is_read', models.BooleanField(default=1)),
                ('status', models.BooleanField(default=1)),
                ('letter', models.ForeignKey(on_delete=django.db.models.deletion.PROTECT, to='letter.letter')),
            ],
            options={
                'db_table': 'tbl_received_letter',
            },
        ),
    ]
