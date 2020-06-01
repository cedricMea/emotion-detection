# pidfile = 'app01.pid' 
reload=False
worker_tmp_dir = '/dev/shm' # worker temp dir 
worker_class = 'gthread' # not understand "gthread"
workers = 2
worker_connections = 1000 # The maximum number of simultaneous clients
timeout = 60 # temps max sans communication afin de killer le worker
keepalive = 2
threads = 4
#proc_name = 'app01'
bind = '0.0.0.0:5001'
backlog = 2048 # nbre maximum de requetes en attentes
accesslog = '-' # log to stout
errorlog = '-' # error to stderr
# user = 'ubuntu'
# group = 'ubuntu'

