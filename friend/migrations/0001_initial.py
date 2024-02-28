# Generated by Django 5.0.2 on 2024-02-28 19:28

import django.db.models.deletion
import django.utils.timezone
from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        ('member', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='Friend',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('created_date', models.DateTimeField(auto_now_add=True)),
                ('updated_date', models.DateTimeField(default=django.utils.timezone.now)),
                ('is_friend', models.SmallIntegerField(default=-1)),
                ('receiver', models.ForeignKey(on_delete=django.db.models.deletion.PROTECT, related_name='friend_receiver', to='member.member')),
                ('sender', models.ForeignKey(on_delete=django.db.models.deletion.PROTECT, related_name='friend_sender', to='member.member')),
            ],
            options={
                'db_table': 'tbl_friend',
            },
        ),
    ]
