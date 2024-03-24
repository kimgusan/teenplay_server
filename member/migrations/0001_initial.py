# Generated by Django 5.0.2 on 2024-03-20 16:17

import django.db.models.deletion
import django.utils.timezone
from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        ('teenplay_server', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='AdminAccount',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('created_date', models.DateTimeField(auto_now_add=True)),
                ('updated_date', models.DateTimeField(default=django.utils.timezone.now)),
                ('admin_id', models.TextField()),
                ('admin_password', models.TextField()),
                ('admin_name', models.TextField()),
            ],
            options={
                'db_table': 'tbl_admin_account',
            },
        ),
        migrations.CreateModel(
            name='Member',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('created_date', models.DateTimeField(auto_now_add=True)),
                ('updated_date', models.DateTimeField(default=django.utils.timezone.now)),
                ('member_email', models.TextField()),
                ('member_nickname', models.TextField()),
                ('member_phone', models.TextField(null=True)),
                ('member_address', models.TextField(null=True)),
                ('member_gender', models.SmallIntegerField(choices=[(0, '선택안함'), (1, '남성'), (2, '여성')], default=0, null=True)),
                ('member_birth', models.IntegerField(null=True)),
                ('member_marketing_agree', models.BooleanField(default=0, null=True)),
                ('member_privacy_agree', models.BooleanField(default=0, null=True)),
                ('status', models.SmallIntegerField(choices=[(-1, '정지'), (0, '탈퇴'), (1, '활동중')], default=1)),
                ('member_type', models.CharField(max_length=10)),
            ],
            options={
                'db_table': 'tbl_member',
            },
        ),
        migrations.CreateModel(
            name='MemberDeleteReason',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('created_date', models.DateTimeField(auto_now_add=True)),
                ('updated_date', models.DateTimeField(default=django.utils.timezone.now)),
                ('delete_reason', models.SmallIntegerField()),
                ('delete_text', models.TextField(null=True)),
            ],
            options={
                'db_table': 'tbl_member_delete_reason',
            },
        ),
        migrations.CreateModel(
            name='MemberFavoriteCategory',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('created_date', models.DateTimeField(auto_now_add=True)),
                ('updated_date', models.DateTimeField(default=django.utils.timezone.now)),
                ('status', models.BooleanField(default=1)),
                ('category', models.ForeignKey(on_delete=django.db.models.deletion.PROTECT, to='teenplay_server.category')),
                ('member', models.ForeignKey(on_delete=django.db.models.deletion.PROTECT, to='member.member')),
            ],
            options={
                'db_table': 'tbl_member_favorite_category',
            },
        ),
        migrations.CreateModel(
            name='MemberProfile',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('created_date', models.DateTimeField(auto_now_add=True)),
                ('updated_date', models.DateTimeField(default=django.utils.timezone.now)),
                ('profile_path', models.ImageField(upload_to='member/%Y/%m/%d')),
                ('status', models.BooleanField(default=1)),
                ('member', models.ForeignKey(on_delete=django.db.models.deletion.PROTECT, to='member.member')),
            ],
            options={
                'db_table': 'tbl_member_profile',
            },
        ),
    ]