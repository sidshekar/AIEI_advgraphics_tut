#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <chrono>

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3native.h>
#define GLM_FORCE_DEFAULT_ALIGNED_GENTYPES
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>
#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <tiny_gltf.h>

#include "Buffer.hpp"
#include "DescriptorSet.hpp"
#include "Image.hpp"
#include "Pipeline.hpp"
#include "RenderGraph.hpp"
#include "RHI.hpp"

#undef max

const float PI = 3.14159265358979323846f;

const uint32_t WIDTH = 800;
const uint32_t HEIGHT = 600;

const uint32_t PARTICLE_GRID_X = 3;
const uint32_t PARTICLE_GRID_Y = 3;
const uint32_t PARTICLE_GRID_Z = 3;
const uint32_t PARTICLE_COUNT = PARTICLE_GRID_X * PARTICLE_GRID_Y * PARTICLE_GRID_Z;

struct Vertex
{
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec2 texCoord;

    static vk::VertexInputBindingDescription getBindingDescription() {
        vk::VertexInputBindingDescription bindingDescription{};
        bindingDescription.binding = 0;
        bindingDescription.stride = sizeof(Vertex);
        return bindingDescription;
    }

    static std::vector<vk::VertexInputAttributeDescription> getAttributeDescriptions() {
        return std::vector<vk::VertexInputAttributeDescription>{
            vk::VertexInputAttributeDescription(0, 0, vk::Format::eR32G32B32Sfloat, offsetof(Vertex, position)),
                vk::VertexInputAttributeDescription(1, 0, vk::Format::eR32G32B32Sfloat, offsetof(Vertex, normal)),
                vk::VertexInputAttributeDescription(2, 0, vk::Format::eR32G32Sfloat, offsetof(Vertex, texCoord))
        };
    }
};

struct Texture
{
    std::vector<uint8_t> imageData;
    int width;
    int height;
};

struct Instance
{
    glm::mat4 model;
    glm::vec3 colour;
    glm::vec3 particleOrbit;
    glm::vec3 particleOffset;
};

struct UniformBufferObject
{
    glm::mat4 view;
    glm::mat4 proj;
    glm::mat4 lightView;
    glm::mat4 lightProj;
    glm::quat rotation;
    glm::vec4 nLightDir;
    uint32_t particleCount;
    float time;
    glm::uvec2 res;
    glm::vec4 sunColor;
    glm::vec4 ambientColor;
};

class HelloTriangleApplication {
public:
    void run() {
        initWindow();
        initVulkan();
        mainLoop();
        cleanup();
    }

private:
    GLFWwindow* window = nullptr;

    Gfx::RHI rhi{};
    Gfx::RenderGraph graph{ rhi };

    std::vector<Vertex> vertices{};
    std::vector<uint32_t> indices{};
    std::vector<Texture> textures{};
    std::vector<vk::DrawIndexedIndirectCommand> drawCmds{};
    std::vector<Instance> instances{};

    std::vector<Gfx::Image> textureImages{};
    Gfx::Sampler textureSampler = nullptr;
    Gfx::Image gbufferAlbedoImage = nullptr;
    Gfx::Image gbufferNormalImage = nullptr;
    Gfx::Image gbufferPositionImage = nullptr;
    Gfx::Image gbufferInstanceIDImage = nullptr;
    Gfx::Sampler gbufferSampler = nullptr;
    Gfx::Image shadowImage = nullptr;
    Gfx::Sampler shadowSampler = nullptr;
    Gfx::Image postprocImage = nullptr;
    Gfx::Sampler postprocSampler = nullptr;
    Gfx::Buffer vertexBuffer = nullptr;
    Gfx::Buffer indexBuffer = nullptr;
    Gfx::Buffer indirectBuffer = nullptr;
    Gfx::Buffer storageBuffer = nullptr;
    Gfx::Buffer uniformBuffer = nullptr;

    void initWindow() {
        glfwInit();

        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

        window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan Renderer", nullptr, nullptr);
    }

    void initVulkan() {
        rhi.init("Vulkan Renderer", getRequiredExtensions(), glfwGetWin32Window(window));

        loadParticles();
        loadFloor();
        loadModel();

        createTextureResources();
        createShadowResources();
        createGBufferResources();
        createPostprocResources();
        createVertexBuffer();
        createIndexBuffer();
        createIndirectBuffer();
        createUniformBuffers();
        createStorageBuffer();

        graph.computePass("Particle", PARTICLE_COUNT)
            .shader("Shaders/particle.comp.spv")
            .shaderBinding(uniformBuffer)
            .shaderBinding(storageBuffer);

        graph.graphicsPass("Shadow")
            .vertexShader("Shaders/shadow.vert.spv")
            .fragmentShader("Shaders/shadow.frag.spv")
            .vertexBuffer<Vertex>(vertexBuffer)
            .indexBuffer(indexBuffer)
            .drawCommandBuffer(indirectBuffer)
            .vertexShaderBinding(uniformBuffer)
            .vertexShaderBinding(storageBuffer)
            .renderTarget(shadowImage);

        graph.graphicsPass("Base")
            .vertexShader("Shaders/gbuffer.vert.spv")
            .fragmentShader("Shaders/gbuffer.frag.spv")
            .vertexBuffer<Vertex>(vertexBuffer)
            .indexBuffer(indexBuffer)
            .drawCommandBuffer(indirectBuffer)
            .allShadersBinding(uniformBuffer)
            .vertexShaderBinding(storageBuffer)
            .fragmentShaderBinding(textureImages, textureSampler)
            .renderTarget(gbufferAlbedoImage)
            .renderTarget(gbufferNormalImage)
            .renderTarget(gbufferPositionImage)
            .renderTarget(gbufferInstanceIDImage)
            .renderTargetSwapChainDepth();

        graph.graphicsPass("Skybox")
            .vertexShader("Shaders/cloud.vert.spv")
            .fragmentShader("Shaders/cloud.frag.spv")
            .fragmentShaderBinding(uniformBuffer)
            .renderTarget(postprocImage);

        graph.graphicsPass("Lighting")
            .vertexShader("Shaders/lighting.vert.spv")
            .fragmentShader("Shaders/lighting.frag.spv")
            .fragmentShaderBinding(uniformBuffer)
            .fragmentShaderBinding(storageBuffer)
            .fragmentShaderBinding(gbufferAlbedoImage, gbufferSampler)
            .fragmentShaderBinding(gbufferNormalImage, gbufferSampler)
            .fragmentShaderBinding(gbufferPositionImage, gbufferSampler)
            .fragmentShaderBinding(shadowImage, shadowSampler)
            .fragmentShaderBinding(gbufferInstanceIDImage)
            .renderTarget(postprocImage);

        graph.graphicsPass("Postprocess")
            .vertexShader("Shaders/postproc.vert.spv")
            .fragmentShader("Shaders/postproc.frag.spv")
            .fragmentShaderBinding(uniformBuffer)
            .fragmentShaderBinding(postprocImage, postprocSampler)
            .renderTargetSwapChainColor();
    }

    std::vector<const char*> getRequiredExtensions() {
        uint32_t glfwExtensionCount = 0;
        auto glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
        return std::vector<const char*>(glfwExtensions, glfwExtensions + glfwExtensionCount);
    }

    std::vector<Vertex> generateSphere(uint32_t latSegments = 8, uint32_t lonSegments = 8) {
        std::vector<Vertex> vertices{};

        for (uint32_t y = 0; y <= latSegments; y++) {
            auto v = float(y) / latSegments;
            auto theta = v * PI;

            for (uint32_t x = 0; x <= lonSegments; x++) {
                auto u = float(x) / lonSegments;
                auto phi = u * 2.0f * PI;

                auto sinTheta = sin(theta);
                auto cosTheta = cos(theta);
                auto sinPhi = sin(phi);
                auto cosPhi = cos(phi);

                glm::vec3 pos{
                    0.5 * sinTheta * cosPhi,
                    0.5 * cosTheta,
                    0.5 * sinTheta * sinPhi
                };

                glm::vec3 normal = glm::normalize(pos);

                glm::vec2 uv{ u, 1.0f - v };

                vertices.push_back({ pos, normal, uv });
            }
        }

        return vertices;
    }

    std::vector<uint32_t> generateSphereIndices(uint32_t latSegments = 8, uint32_t lonSegments = 8) {
        std::vector<uint32_t> indices{};

        for (uint32_t y = 0; y < latSegments; y++) {
            for (uint32_t x = 0; x < lonSegments; x++) {
                auto i0 = y * (lonSegments + 1) + x;
                auto i1 = i0 + 1;
                auto i2 = i0 + (lonSegments + 1);
                auto i3 = i2 + 1;

                indices.emplace_back(i0);
                indices.emplace_back(i1);
                indices.emplace_back(i2);

                indices.emplace_back(i1);
                indices.emplace_back(i3);
                indices.emplace_back(i2);
            }
        }

        return indices;
    }

    void loadParticles() {
        auto sphere = generateSphere();
        auto sphereIndices = generateSphereIndices();

        vk::DrawIndexedIndirectCommand drawCmd{
            static_cast<uint32_t>(sphereIndices.size()), // index count
            PARTICLE_COUNT, // instance count
            static_cast<uint32_t>(indices.size()), // first index
            static_cast<int32_t>(vertices.size()), // vertex offset
            static_cast<uint32_t>(instances.size()) // first instance
        };

        drawCmds.emplace_back(std::move(drawCmd));

        vertices.insert(vertices.end(), sphere.begin(), sphere.end());
        indices.insert(indices.end(), sphereIndices.begin(), sphereIndices.end());

        Texture texture{ { 255, 255, 255, 255 }, 1, 1 }; // white 1x1 texture

        textures.resize(textures.size() + PARTICLE_COUNT, std::move(texture));

        std::mt19937 rng(std::random_device{}());

        glm::vec3 center{ -1, 0, 0.5 };

        for (uint32_t i = 0; i < PARTICLE_GRID_X; i++) {
            for (uint32_t j = 0; j < PARTICLE_GRID_Y; j++) {
                for (uint32_t k = 0; k < PARTICLE_GRID_Z; k++) {
                    float f = 0.1f;
                    float x = i * f, y = j * f, z = k * f;
                    auto nz = std::uniform_real_distribution<float>{ -1, 1 }(rng);
                    auto nt = std::uniform_real_distribution<float>{ 0, 2 * PI }(rng);
                    auto nr = sqrtf(1.0f - z * z);
                    auto orbit = std::uniform_real_distribution<float>{ 0.125, 0.25 }(rng);
                    auto scale = std::uniform_real_distribution<float>{ 0.03125, 0.0625 }(rng);
                    auto r = std::uniform_real_distribution<float>{ 0, 1 }(rng);
                    auto g = std::uniform_real_distribution<float>{ 0, 1 }(rng);
                    auto b = std::uniform_real_distribution<float>{ 0, 1 }(rng);

                    Instance instance{};
                    instance.model = glm::translate(glm::mat4(1.0f), center + glm::vec3(x, y, z)) * glm::scale(glm::mat4(1.0f), glm::vec3(scale));
                    instance.colour = glm::vec3(r, g, b);
                    instance.particleOrbit = glm::vec3(nr * cosf(nt), nr * sinf(nt), nz) * orbit;

                    instances.emplace_back(std::move(instance));
                }
            }
        }
    }

    void loadFloor() {
        std::vector<Vertex> quad{
            {{-0.5, -0.5, 0.0}, {0.0, 0.0, 1.0}, {1.0, 0.0}},
            {{0.5, -0.5, 0.0}, {0.0, 0.0, 1.0}, {0.0, 0.0}},
            {{0.5, 0.5, 0.0}, {0.0, 0.0, 1.0}, {0.0, 1.0}},
            {{-0.5, 0.5, 0.0}, {0.0, 0.0, 1.0}, {1.0, 1.0}}
        };

        std::vector<uint32_t> quadIndices{
            0, 1, 2, 2, 3, 0
        };

        vk::DrawIndexedIndirectCommand drawCmd{
            static_cast<uint32_t>(quadIndices.size()), // index count
            1, // instance count
            static_cast<uint32_t>(indices.size()), // first index
            static_cast<int32_t>(vertices.size()), // vertex offset
            static_cast<uint32_t>(instances.size()) // first instance
        };

        drawCmds.emplace_back(std::move(drawCmd));

        vertices.insert(vertices.end(), quad.begin(), quad.end());
        indices.insert(indices.end(), quadIndices.begin(), quadIndices.end());

        Texture texture{};

        int texChannels;
        auto pixels = stbi_load("Textures/statue.jpg", &texture.width, &texture.height, &texChannels, STBI_rgb_alpha);

        if (!pixels) {
            throw std::runtime_error("failed to load texture image!");
        }

        texture.imageData.resize(texture.width * texture.height * 4);
        memcpy(texture.imageData.data(), pixels, texture.imageData.size());

        stbi_image_free(pixels);

        textures.emplace_back(std::move(texture));

        Instance instance{};
        instance.model = glm::translate(glm::mat4(1.0f), glm::vec3(0.0, 0.0, -0.5)) * glm::scale(glm::mat4(1.0f), glm::vec3(4.0));
        instance.colour = glm::vec3(1.0f, 1.0f, 0.0f);

        instances.emplace_back(std::move(instance));
    }

    template<typename T>
    std::vector<T> ReadAccessor(const tinygltf::Model& model, const tinygltf::Accessor& accessor) {
        auto& view = model.bufferViews[accessor.bufferView];
        auto& buffer = model.buffers[view.buffer];

        auto dataPtr = buffer.data.data() + view.byteOffset + accessor.byteOffset;
        auto stride = accessor.ByteStride(view);
        auto count = accessor.count;

        std::vector<T> out(count);

        if (stride == sizeof(T)) {
            // tightly packed
            memcpy(out.data(), dataPtr, count * sizeof(T));
        }
        else {
            // interleaved
            for (size_t i = 0; i < count; i++) {
                memcpy(&out[i], dataPtr + i * stride, sizeof(T));
            }
        }

        return out;
    }

    uint32_t LoadPrimitive(const tinygltf::Model& model, const tinygltf::Primitive& primitive) {
        const tinygltf::Accessor& posAcc =
            model.accessors[primitive.attributes.at("POSITION")];
        auto positions = ReadAccessor<glm::vec3>(model, posAcc);

        std::vector<glm::vec3> normals;
        if (primitive.attributes.count("NORMAL")) {
            auto& normalAcc = model.accessors[primitive.attributes.at("NORMAL")];
            normals = ReadAccessor<glm::vec3>(model, normalAcc);
        }
        else {
            normals.resize(positions.size(), glm::vec3(0));
        }

        std::vector<glm::vec2> texCoords;
        if (primitive.attributes.count("TEXCOORD_0")) {
            auto& texCoordAcc = model.accessors[primitive.attributes.at("TEXCOORD_0")];
            texCoords = ReadAccessor<glm::vec2>(model, texCoordAcc);
        }
        else {
            texCoords.resize(positions.size(), glm::vec2(0));
        }

        vertices.reserve(vertices.size() + positions.size());
        for (size_t i = 0; i < positions.size(); i++) {
            vertices.emplace_back(Vertex{ positions[i], normals[i], texCoords[i] });
        }

        auto& idxAcc = model.accessors[primitive.indices];
        auto& view = model.bufferViews[idxAcc.bufferView];
        auto& buffer = model.buffers[view.buffer];

        auto dataPtr = buffer.data.data() + view.byteOffset + idxAcc.byteOffset;

        auto count = idxAcc.count;

        indices.reserve(indices.size() + count);

        switch (idxAcc.componentType) {
        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT:
        {
            auto src = reinterpret_cast<const uint16_t*>(dataPtr);
            for (size_t i = 0; i < count; i++) {
                indices.emplace_back(src[i]);
            }
            break;
        }
        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT:
        {
            auto src = reinterpret_cast<const uint32_t*>(dataPtr);
            for (size_t i = 0; i < count; i++) {
                indices.emplace_back(src[i]);
            }
            break;
        }
        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:
        {
            auto src = reinterpret_cast<const uint8_t*>(dataPtr);
            for (size_t i = 0; i < count; i++) {
                indices.emplace_back(src[i]);
            }
            break;
        }
        default:
            throw std::runtime_error("Unsupported index type");
        }

        return count;
    }

    void loadModel() {
        tinygltf::TinyGLTF loader{};

        tinygltf::Model model;
        std::string err;
        if (!loader.LoadASCIIFromFile(&model, &err, nullptr, "Models/CesiumMan.gltf")) {
            throw std::runtime_error("Failed to load model: " + err);
        }

        for (auto& mesh : model.meshes) {
            for (auto& primitive : mesh.primitives) {
                vk::DrawIndexedIndirectCommand drawCmd{
                    0, // index count
                    1, // instance count
                    static_cast<uint32_t>(indices.size()), // first index
                    static_cast<int32_t>(vertices.size()), // vertex offset
                    static_cast<uint32_t>(instances.size()) // first instance
                };

                drawCmd.indexCount = LoadPrimitive(model, primitive);

                drawCmds.emplace_back(std::move(drawCmd));
            }
        }

        Texture texture{};

        int texChannels;
        auto pixels = stbi_load("Models/CesiumMan_img0.jpg", &texture.width, &texture.height, &texChannels, STBI_rgb_alpha);

        if (!pixels) {
            throw std::runtime_error("failed to load texture image!");
        }

        texture.imageData.resize(texture.width * texture.height * 4);
        memcpy(texture.imageData.data(), pixels, texture.imageData.size());

        stbi_image_free(pixels);

        textures.emplace_back(std::move(texture));

        Instance instance{};
        instance.model = glm::translate(glm::mat4(1.0f), glm::vec3(0.0, 0.0, -0.5));
        instance.colour = glm::vec3(1.0f, 1.0f, 1.0f);

        instances.emplace_back(std::move(instance));
    }

    void createTextureResources() {
        textureImages.reserve(textures.size());

        for (auto& texture : textures) {
            auto textureImage = rhi.createImage2D(vk::Format::eR8G8B8A8Srgb,
                vk::Extent2D(texture.width, texture.height),
                vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled);

            rhi.updateImage(textureImage, texture.imageData);
            textureImages.emplace_back(std::move(textureImage));
        }

        vk::SamplerCreateInfo samplerInfo{};
        samplerInfo.magFilter = vk::Filter::eLinear;
        samplerInfo.minFilter = vk::Filter::eLinear;
        samplerInfo.mipmapMode = vk::SamplerMipmapMode::eLinear;
        samplerInfo.anisotropyEnable = true;

        textureSampler = rhi.createSampler(samplerInfo);
    }

    void createShadowResources() {
        shadowImage = rhi.createImage2D(rhi.getDepthFormat(),
            vk::ImageUsageFlagBits::eDepthStencilAttachment | vk::ImageUsageFlagBits::eSampled);

        vk::SamplerCreateInfo samplerInfo{};
        samplerInfo.magFilter = vk::Filter::eLinear;
        samplerInfo.minFilter = vk::Filter::eLinear;
        samplerInfo.mipmapMode = vk::SamplerMipmapMode::eLinear;

        shadowSampler = rhi.createSampler(samplerInfo);
    }

    void createPostprocResources() {
        postprocImage = rhi.createImage2D(rhi.getSurfaceFormat(),
            vk::ImageUsageFlagBits::eColorAttachment);

        vk::SamplerCreateInfo samplerInfo{};
        samplerInfo.magFilter = vk::Filter::eLinear;
        samplerInfo.minFilter = vk::Filter::eLinear;
        samplerInfo.mipmapMode = vk::SamplerMipmapMode::eLinear;

        postprocSampler = rhi.createSampler(samplerInfo);
    }

    void createVertexBuffer() {
        vk::BufferCreateInfo bufferInfo{};
        bufferInfo.size = sizeof(vertices[0]) * vertices.size();
        bufferInfo.usage = vk::BufferUsageFlagBits::eVertexBuffer;

        vertexBuffer = rhi.createBuffer(bufferInfo, vertices);
    }

    void createIndexBuffer() {
        vk::BufferCreateInfo bufferInfo{};
        bufferInfo.size = sizeof(indices[0]) * indices.size();
        bufferInfo.usage = vk::BufferUsageFlagBits::eIndexBuffer;

        indexBuffer = rhi.createBuffer(bufferInfo, indices);
    }

    void createIndirectBuffer() {
        vk::BufferCreateInfo bufferInfo{};
        bufferInfo.size = sizeof(drawCmds[0]) * drawCmds.size();
        bufferInfo.usage = vk::BufferUsageFlagBits::eIndirectBuffer;

        indirectBuffer = rhi.createBuffer(bufferInfo, drawCmds);
    }

    void createUniformBuffers() {
        vk::BufferCreateInfo bufferInfo{};
        bufferInfo.size = sizeof(UniformBufferObject);
        bufferInfo.usage = vk::BufferUsageFlagBits::eUniformBuffer;

        uniformBuffer = rhi.createBuffer(bufferInfo,
            vk::MemoryPropertyFlagBits::eHostVisible |
            vk::MemoryPropertyFlagBits::eHostCoherent);

        uniformBuffer.map();
    }

    void createStorageBuffer() {
        vk::BufferCreateInfo bufferInfo{};
        bufferInfo.size = sizeof(instances[0]) * instances.size();
        bufferInfo.usage = vk::BufferUsageFlagBits::eStorageBuffer;

        storageBuffer = rhi.createBuffer(bufferInfo, instances);
    }

    void createGBufferResources() {
        gbufferAlbedoImage = rhi.createImage2D(rhi.getSurfaceFormat(),
            vk::ImageUsageFlagBits::eColorAttachment);
        gbufferNormalImage = rhi.createImage2D(vk::Format::eR16G16B16A16Sfloat,
            vk::ImageUsageFlagBits::eColorAttachment);
        gbufferPositionImage = rhi.createImage2D(vk::Format::eR32G32B32A32Sfloat,
            vk::ImageUsageFlagBits::eColorAttachment);
        gbufferInstanceIDImage = rhi.createImage2D(vk::Format::eR32Uint,
            vk::ImageUsageFlagBits::eColorAttachment);

        vk::SamplerCreateInfo samplerInfo{};
        samplerInfo.magFilter = vk::Filter::eNearest;
        samplerInfo.minFilter = vk::Filter::eNearest;
        samplerInfo.addressModeU = vk::SamplerAddressMode::eClampToEdge;
        samplerInfo.addressModeV = vk::SamplerAddressMode::eClampToEdge;

        gbufferSampler = rhi.createSampler(samplerInfo);
    }

    void updateUniformBuffer(uint32_t currentImage) {
        static auto startTime = std::chrono::high_resolution_clock::now();

        auto currentTime = std::chrono::high_resolution_clock::now();
        float time = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();

        auto swapChainExtent = rhi.getSwapChainExtent();

        UniformBufferObject ubo{};
        ubo.view = lookAt(glm::vec3(2.0f, 2.0f, 2.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
        ubo.proj = glm::perspective(glm::radians(45.0f), static_cast<float>(swapChainExtent.width) / static_cast<float>(swapChainExtent.height), 0.1f, 10.0f);
        ubo.proj[1][1] *= -1;
        ubo.rotation = glm::angleAxis(time * glm::radians(90.0f), glm::vec3(0.0f, 0.0f, 1.0f));
        auto nLightDir = -glm::normalize(glm::vec3(-1.0f, 1.0, -1.0));
        ubo.nLightDir = glm::vec4(nLightDir, 0.0f);
        ubo.lightView = lookAt(nLightDir, glm::vec3(0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
        ubo.lightProj = glm::ortho(-3.0f, 3.0f, -3.0f, 3.0f, 0.1f, 10.0f);
        ubo.lightProj[1][1] *= -1;
        ubo.particleCount = PARTICLE_COUNT;
        ubo.time = time;
        ubo.res.x = swapChainExtent.width;
        ubo.res.y = swapChainExtent.height;
        ubo.sunColor = glm::vec4(1.0f, 0.95f, 0.85f, 0.0f);
        ubo.ambientColor = glm::vec4(0.3f, 0.5f, 0.8f, 0.0f);

        memcpy(uniformBuffer.getMappedData(currentImage), &ubo, sizeof(ubo));
    }

    void drawFrame() {
        updateUniformBuffer(static_cast<uint32_t>(graph.getFrameIndex()));
        graph.executeFrame();
    }

    void mainLoop() {
        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();
            drawFrame();
        }

        rhi.getDevice().waitIdle();
    }

    void cleanup() {
        glfwDestroyWindow(window);

        glfwTerminate();
    }
};

int main() {
    try {
        HelloTriangleApplication app;
        app.run();
    }
    catch (const std::exception& e) {
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
