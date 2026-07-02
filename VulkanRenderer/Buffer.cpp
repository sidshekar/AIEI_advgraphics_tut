#include "Buffer.hpp"

using Gfx::Buffer;

Buffer::Buffer(
	const vk::BufferCreateInfo& createInfo, 
	std::vector<vk::raii::Buffer>&& buffers, 
	std::vector<vk::raii::DeviceMemory>&& bufferMemories):
	m_createInfo(std::move(createInfo)),
	m_buffers(std::move(buffers)), 
	m_bufferMemories(std::move(bufferMemories))
{
	m_mappedData.resize(m_buffers.size());

	m_info.createInfo = m_createInfo;
	m_info.buffers.reserve(m_buffers.size());

	for (const auto& buffer : m_buffers)
	{
		m_info.buffers.emplace_back(*buffer);
	}
}

void Buffer::map()
{
	for (size_t i = 0; i < m_bufferMemories.size(); i++)
	{
		m_mappedData[i] = m_bufferMemories[i].mapMemory(0, m_createInfo.size);
	}
}

void Buffer::unmap()
{
	for (size_t i = 0; i < m_bufferMemories.size(); i++)
	{
		m_bufferMemories[i].unmapMemory();
		m_mappedData[i] = nullptr;
	}
}