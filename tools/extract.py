import tensorflow as tf
import utils
import json

def showValue(sess, graph, feed_dict, path):
  m_tensor = graph.get_tensor_by_name(path)
  val = sess.run(m_tensor, feed_dict = feed_dict)
  
  print val
  print val.shape
  
  return val

with open("vgg16.tfmodel", mode='rb') as f:
  fileContent = f.read()

graph_def = tf.GraphDef()
graph_def.ParseFromString(fileContent)

images = tf.placeholder("float", [None, 224, 224, 3])

tf.import_graph_def(graph_def, input_map={ "images": images })
print "graph loaded from disk"

graph = tf.get_default_graph()

# get nodes value  
cat = utils.load_image("1771292.jpg")

with tf.Session() as sess:
  init = tf.initialize_all_variables()
  sess.run(init)
  print "variables initialized"
  
  batch = cat.reshape((1, 224, 224, 3))
  assert batch.shape == (1, 224, 224, 3)

  feed_dict = { images: batch }

  import_mul_y_tensor = graph.get_tensor_by_name("import/fc7/BiasAdd:0")
  import_mul_y = sess.run(import_mul_y_tensor, feed_dict = feed_dict)
  
  print import_mul_y
  print import_mul_y.shape
  
  '''
  # print classification result
  prob_tensor = graph.get_tensor_by_name("import/prob:0")
  prob = sess.run(prob_tensor, feed_dict=feed_dict)
  
  utils.print_prob(prob[0])
  '''
  
  '''
  # print operations
  m_ops = graph.get_operations()
  print json.dumps([m_op.name for m_op in m_opssh], indent=1)
  
  summar_writer = tf.train.SummaryWriter('./log', sess.graph)
  
  for m_op in m_ops: tf.scalar_summary(m_op.name, m_op.outputs)

  summaries = tf.merge_all_summaries()
  '''
  
  # summar_writer.add_summary(sess.run(summaries), 0)
  # print prob_2 - prob
  # print prob_2.shape
  # print prob[prob != 0]
  # print prob.shape
  # print (batch - batch_2)
  

