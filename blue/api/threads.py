
rows = [('pid1','guest1','target1'),('pid2','guest2','target2'),('pid3','guest3','target3')]
record={}

for each_record in rows:
    print(each_record)
    prob={}
    guest_ids = each_record[1]
    print(guest_ids)
    guest_target = each_record[2]
    print(guest_target)
    prob[guest_ids]=guest_target
    record[each_record[0]]=prob

print("RECORD",record)    