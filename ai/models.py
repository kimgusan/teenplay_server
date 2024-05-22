from django.db import models


class ReplyAi(models.Model):
    comment = models.TextField(null=False, blank=False)
    target = models.SmallIntegerField(null=False, blank=False)

    class Meta:
        db_table = 'tbl_reply_ai'