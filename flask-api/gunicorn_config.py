# pidfile = 'app01.pid' 
# reload=False
worker_tmp_dir = '/dev/shm' # worker temp dir 
worker_class = 'gthread' # not understand "gthread"
# workers = 1
# worker_connections = 10 # The maximum number of simultaneous clients
timeout = 1000 # temps max sans communication afin de killer le worker
# keepalive = 2
# threads = 2
# proc_name = 'app01'
bind = '0.0.0.0:5000'
# backlog = 2048 # nbre maximum de requetes en attentes
accesslog = '-' # log to stout
errorlog = '-' # error to stderr
# user = 'ubuntu'
# group = 'ubuntu'

