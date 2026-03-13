from pymilvus import MilvusClient                                                                                                                                                                                                                                             

print("######### test")

client = MilvusClient(uri='http://localhost:19530')                                                                                                                                                                                                                           
results = client.query('rag_docs', filter='id > 0', output_fields=['id', 'text'])                                                                                                                                                                                             
for r in results:                                                                                                                                                                                                                                                             
    print(r['id'], r['text'][:80]) 
