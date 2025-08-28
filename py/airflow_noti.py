from airflow.decorators import dag, task
from airflow.models.baseoperator import chain
from airflow.exceptions import AirflowFailException
from airflow.providers.apprise.notifications.apprise import send_apprise_notification
from apprise import NotifyType

@dag(schedule=None, catchup=False, on_failure_callback=send_apprise_notification(
    title='Airflow Task Failed',
    body='Task {{ task_instance_key_str }} failed',
    notify_type=NotifyType.FAILURE,
    apprise_conn_id='notifier',
    tag='alerts'
))
def my_dag():
    
    @task
    def a():
        print('good')
        
    @task
    def b():
        print('bad')
        raise AirflowFailException()
    
    chain(a(), b())
    
my_dag()