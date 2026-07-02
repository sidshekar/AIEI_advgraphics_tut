#include "Image.hpp"

using Gfx::Image;
using Gfx::Sampler;

Image::Image(
    const vk::ImageCreateInfo& createInfo, 
    std::vector<vk::raii::Image>&& images, 
    std::vector<vk::raii::DeviceMemory>&& imageMemories, 
    std::vector<vk::raii::ImageView>&& imageViews):
    m_createInfo(createInfo),
    m_images(std::move(images)),
    m_imageMemories(std::move(imageMemories)),
	m_imageViews(std::move(imageViews))
{
    m_info.createInfo = m_createInfo;
    m_info.images.reserve(m_images.size());
    m_info.imageViews.reserve(m_imageViews.size());

    for (size_t i = 0; i < m_images.size(); i++)
    {
        m_info.images.emplace_back(*m_images[i]);
        m_info.imageViews.emplace_back(*m_imageViews[i]);
    }
}

Sampler::Sampler(const vk::SamplerCreateInfo& createInfo, vk::raii::Sampler&& sampler):
    m_sampler(std::move(sampler))
{}