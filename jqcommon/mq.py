# coding: utf-8

import time
import json
import logging

import pika

from retrying import retry
from .service import get_mq_channel


class MQClient(object):
    """Message queue client"""

    def __init__(self, mq_urls=None, routing_key='', msg_handler=None, debug=True):
        self._mq_urls = mq_urls
        self._routing_key = routing_key
        self._msg_handler = msg_handler

        # 接收任务的 consumer tag, use to cancel that consumer, returned by channel.basic_consume()
        self._task_consumer_tag = None

        # pika Connection
        self._con = None
        # pika Channel
        self._channel = None

        self.log = logging.getLogger("jqcommon.mq.MQClient")

    def run(self):
        while 1:
            try:
                self._setup()
                self._loop()
                break
            except pika.exceptions.AMQPError:
                self.log.info('connection error. reconnecting...')
                time.sleep(3)
            finally:
                self._cleanup()

    @property
    def _queue_name_suffix(self):
        return '_' + self._routing_key if self._routing_key else ''

    @property
    def routing_key(self):
        return self._routing_key

    def _setup(self):
        self._channel = get_mq_channel(self._mq_urls)
        self._con = self._channel.connection
        self._channel.confirm_delivery()
        self._channel.basic_qos(prefetch_count=1, all_channels=True)

    def _loop(self):
        self.log.info("run loop...")
        while self._channel is not None:
            self.enable_consuming()

            self.log.info("start_consuming")
            self._channel.start_consuming()

        self.log.info("exit loop...")

    def _cleanup(self):
        self.log.info("clean up")
        if self._con is not None:
            if not self._con.is_closed:
                self._con.close()
            self._con = None
            self._channel = None

    def enable_consuming(self):
        """ 开始接收任务 """
        self._task_consumer_tag = self._setup_consumer(
            self._msg_handler,
            queue='queue_live_middle_tier' + self._queue_name_suffix,
            exchange='direct_exchange_live_middle_tier',
            exchange_type='direct')

        self.log.info("enable_consuming, consumer tag: {}".format(self._task_consumer_tag))

    def _setup_consumer(self, msg_callback, queue, exchange, exchange_type):
        """ Setup consumer, run msg_callback(msg) when receive msg on queue `queue`
        on exchange `exchange`
        """
        def consume_callback(ch, method, properties, body):
            self.log.info("receive msg %r on queue %r exchange %r" %
                          (body, queue, exchange))
            try:
                self._handle_consume_callback(body, msg_callback, exchange, queue)
            except Exception:
                self.log.exception("handle msg callback failed")
            self._channel.basic_ack(method.delivery_tag)

        self._channel.exchange_declare(
            exchange=exchange, durable=True, exchange_type=exchange_type)
        if queue is not None:
            self._channel.queue_declare(queue=queue, durable=True)
        else:
            result = self._channel.queue_declare(exclusive=True)
            queue = result.method.queue

        self._channel.queue_bind(exchange=exchange,
                                 queue=queue,
                                 routing_key=self._routing_key)
        self.log.info("consume on exchange: {exchange}, queue: {queue}, routing_key: {routing_key}".format(
            exchange=exchange,
            queue=queue,
            routing_key=self._routing_key))
        return self._channel.basic_consume(consume_callback, queue=queue, no_ack=False)

    def _handle_consume_callback(self, msg, msg_callback, exchange, queue):
        """ Called when each msg come, run msg_callback """
        if self._handle_heartbeat(msg, exchange, queue):
            return

        msg_callback(msg)

    def _handle_heartbeat(self, msg, exchange, queue):
        """ 我们会有脚本检测 exchange/queue 是否 ok, 这个函数是回应这个检查的, return True if msg is a heartbeat message """
        import datetime
        import json
        import socket

        if not msg.startswith('-'):
            return False

        self.log.info("handle heartbeat: {} on exchange {} queue {}".format(msg, exchange, queue))
        self._channel.queue_declare(queue='queue_ack')
        self._channel.basic_publish(exchange='', routing_key='queue_ack', body=json.dumps({
            'exchange': exchange,
            'queue': queue,
            'body': msg,
            'time': str(datetime.datetime.now()),
            'host': socket.gethostname(),
        }))

        return True

    @retry(stop_max_attempt_number=600, wait_random_min=1000, wait_random_max=5000)
    def _publish_msg(self, msg, exchange=None, routing_key=''):
        exchange = "direct_exchange_live_middle_tier" if exchange is None else exchange
        result = self._channel.basic_publish(exchange=exchange,
                                             routing_key=routing_key, body=msg)

        out_info = "send {} to {}, routing_key={}, result:{}".format(msg,
                                                                     exchange, routing_key, result)
        if result:
            self.log.info(out_info)
        else:
            raise Exception("Error " + out_info)

    def send_message(self, type, exchange=None, routing_key='', **extra_fields):
        msg = {"type": type}
        msg.update(extra_fields)
        self._publish_msg(msg=json.dumps(msg), exchange=exchange, routing_key=routing_key)
