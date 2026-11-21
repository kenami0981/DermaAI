using Derma.Application.SkinAnalysis;
using Derma.Domain;
using Derma.Infrastructure;
using MediatR;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using Derma.API.Models;
using Microsoft.AspNetCore.Authorization;
using System.Security.Claims;

namespace Derma.API.Controllers
{
    
    public class SkinAnalysisController : BaseApiController
    {
        private readonly IMediator _mediator;
        private Guid CurrentUserId =>
    Guid.TryParse(User.FindFirstValue(ClaimTypes.NameIdentifier), out var id) ? id : Guid.Empty;
        private bool IsAdmin => User.IsInRole("Admin");
        public SkinAnalysisController(IMediator mediator)
        {
            _mediator = mediator;
        }


        [HttpGet]
        [Authorize(AuthenticationSchemes = "Bearer,ApiKey")]
        public async Task<ActionResult<List<SkinAnalysis>>> GetSkinAnalysis()
        {
            return await Mediator.Send(new SkinAnalysisList.Query { UserId = CurrentUserId, IsAdmin = IsAdmin });
        }

        [HttpGet("{id}")]
        [Authorize(AuthenticationSchemes = "Bearer,ApiKey")]
        public async Task<ActionResult<SkinAnalysis>> GetSkinAnalysis(Guid id)
        {
            var result = await Mediator.Send(new SkinAnalysisDetails.Query
            { Id = id, UserId = CurrentUserId, IsAdmin = IsAdmin });
            return result == null ? NotFound() : result;
        }
        [HttpPost]
        [Authorize]
        [RequestSizeLimit(10_000_000)]
        public async Task<IActionResult> CreateSkinAnalysis([FromForm] SkinAnalysisCreateRequest request)
        {
            var allowed = new[] { ".jpg", ".jpeg", ".png" };
            var extension = Path.GetExtension(request.Image.FileName).ToLowerInvariant();

            if (!allowed.Contains(extension))
                return BadRequest("Dozwolone formaty: jpg, jpeg, png.");

            var folder = Path.Combine(Directory.GetCurrentDirectory(), "Uploads");
            Directory.CreateDirectory(folder);

            var fileName = $"{Guid.NewGuid()}{extension}";
            var path = Path.Combine(folder, fileName);

            using (var stream = new FileStream(path, FileMode.Create))
            {
                await request.Image.CopyToAsync(stream);
            }

            var analysisId = Guid.NewGuid();

            var skinAnalysis = new Derma.Domain.SkinAnalysis
            {
                Id = analysisId,
                userId = Guid.Empty,
                notes = request.notes,
                imageUrl = fileName,
                uploadDate = request.uploadDate,
                Status = AnalysisStatus.Pending,
                Tags = (request.tags ?? new())
                    .Distinct()
                    .Select(t => new SkinAnalysisTag
                    {
                        Id = Guid.NewGuid(),
                        Tag = t
                    })
                    .ToList()
            };

            
            try
            {
                await Mediator.Send(new SkinAnalysisCreate.Command { SkinAnalysis = skinAnalysis });
            }
            catch
            {
                if (System.IO.File.Exists(path))
                    System.IO.File.Delete(path);

                throw;
            }

            return Ok(analysisId);

            
        }
        [HttpPut("{id}")]
        [Authorize]
        public async Task<IActionResult> EditSkinAnalysis(Guid id, SkinAnalysis skinAnalysis)
        {
            skinAnalysis.Id = id;
            await Mediator.Send(new SkinAnalysisEdit.Command { SkinAnalysis = skinAnalysis, UserId = CurrentUserId, IsAdmin = IsAdmin });
            return Ok();
        }
        [HttpDelete("{id}")]
        [Authorize]
        public async Task<IActionResult> DeleteSkinAnalysis(Guid id)
        {
            await Mediator.Send(
                new SkinAnalysisDelete.Command { Id = id , UserId = CurrentUserId, IsAdmin = IsAdmin });

            return Ok();
        }
    }
}
