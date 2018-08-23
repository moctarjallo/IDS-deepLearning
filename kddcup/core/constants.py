training_file = '/home/mctrjalloh/.kddcup/kddcup.data_10_percent_corrected'
testing_file = '/home/mctrjalloh/.kddcup/corrected'
names_file = '/home/mctrjalloh/.kddcup/kddcup.names.txt'

kddcup_properties = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes', \
                     'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot', \
                     'num_failed_logins', 'logged_in', 'num_compromised', \
                     'root_shell', 'su_attempted', 'num_root', 'num_file_creations', \
                     'num_shells', 'num_access_files', 'num_outbound_cmds', \
                     'is_host_login', 'is_guest_login', 'count', 'srv_count', \
                     'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', \
                     'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', \
                     'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', \
                     'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', \
                     'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', \
                     'dst_host_srv_serror_rate', 'dst_host_rerror_rate', \
                     'dst_host_srv_rerror_rate', 'attack_type']

kddcup_targets = ['smurf.', 'multihop.', 'rootkit.', 'phf.', 'neptune.',
                  'satan.', 'land.', 'guess_passwd.', 'warezmaster.', 
                  'ftp_write.', 'teardrop.', 'loadmodule.', 'imap.', 'spy.', 
                  'ipsweep.', 'pod.','buffer_overflow.', 'warezclient.', 
                  'normal.', 'nmap.', 'portsweep.', 'perl.', 'back.']
