import json
import warnings
from typing import Optional

import requests
from requests.auth import HTTPBasicAuth

from airflow.exceptions import AirflowException
from airflow.providers.http.hooks.http import HttpHook

# channel, username, icon_emoji, icon_url, link_names, block
class TeamshookHook(HttpHook):
    def __init__(
        self,
        http_conn_id=None,
        title='',
        title_text='',
        activityTitle='',
        activitySubtitle='',
        message='',
        color='green',
        *args,
        **kwargs,
    ):
        print(http_conn_id)
        super().__init__(http_conn_id=http_conn_id, *args, **kwargs)
        self.conn = self.get_connection(http_conn_id)
        self.teams_url = self.conn.host
        self.title = title
        self.title_text = title_text
        self.activityTitle = activityTitle
        self.activitySubtitle = activitySubtitle
        self.message = message
        self.color = color
        self.color_dict = {'red': 'FF0000', 'green': '00FF00', 'blue': '0000FF'}
        

    def _build_teams_message(self) -> str:
        return {
            "themeColor": self.color_dict[self.color],
            "title": self.title,
            "text": self.title_text,
            "sections" : [
                {
                    "activityTitle": self.activityTitle,
                    "activitySubtitle": self.activitySubtitle,
                    "activityImage":"",
                    "text": self.message
                }
            ]
        }

    def execute(self) -> None:
        teams_message = self._build_teams_message()
        # self.run(
        #     endpoint='',
        #     data=teams_message,
        #     headers = {"cache-control": "no-cache"},
        #     extra_options={'check_response': False},
        # )

        headers = {"cache-control": "no-cache"}
 
        r = requests.post(self.teams_url, json=teams_message, headers=headers)
# TeamsWebhookOperator
# SimpleHttpOperator를 상속받아 구현했으며, TeamsWebhookOperator는 TeamshookHook을 사용하기 위해 parameter을 전달받고, 메세지를 생성해 전달하는 방식이다.

# https://github.com/dydwnsekd/airflow_example/blob/main/customOperator/TeamsWebhookOperator.py

from typing import Any, Dict, Optional

from airflow.providers.http.operators.http import SimpleHttpOperator
from airflow.utils.decorators import apply_defaults
from custom_operator.teams.TeamsHook import TeamshookHook

class TeamsWebhookOperator(SimpleHttpOperator):

    @apply_defaults
    def __init__(
        self,
        *,
        http_conn_id: str,
        title: str='',
        title_text: str='',
        activityTitle: str='',
        activitySubtitle: str='',
        message: str = '',
        color: str = 'green',
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.http_conn_id = http_conn_id
        self.title = title
        self.title_text = title_text
        self.activityTitle = activityTitle
        self.activitySubtitle = activitySubtitle
        self.message = message
        self.color = color
        self.hook: Optional[TeamshookHook] = None

    def execute(self, context: Dict[str, Any]) -> None:
        self.hook = TeamshookHook(
            self.http_conn_id,
            self.title,
            self.title_text,
            self.activityTitle,
            self.activitySubtitle,
            self.message,
            self.color,
        )
        self.hook.execute()




# -*- coding:utf-8 -*-
from airflow import DAG
from datetime import datetime, timedelta
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
import requests
from requests.auth import HTTPBasicAuth

from customOperator.teams.TeamsOperator import TeamsWebhookOperator
 
default_args= {
    'retries': 0,
    'catchup': False,
    'retry_delaty': timedelta(minutes=5),
}
 
dag = DAG(
    'webhook_test_dydwnsekd',
    default_args=default_args,
    start_date=days_ago(1),
    schedule_interval="@once"
)
 
t1 = TeamsWebhookOperator(
    task_id='teams_test',
    http_conn_id='webhook_dydwnsekd',
    title='webhook title!',
    title_text='안녕하세요',
    message='메세지 보내기 성공!',
    color='green',
    dag=dag,
)
 
t1