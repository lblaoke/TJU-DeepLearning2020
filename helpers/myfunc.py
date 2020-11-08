from xml.etree.cElementTree import parse

#find device with most free memory
def free_device_id(path):
	xml_root=parse(path).getroot()
	device_id=0
	max_free_memory=0
	for _id in range(int(xml_root[3].text)):
		xml_text=xml_root[4+_id][23][2].text
		free_memory=int(xml_text.split()[0])

		if free_memory>max_free_memory:
			device_id=_id
			max_free_memory=free_memory

	return device_id
