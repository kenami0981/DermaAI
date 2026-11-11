using Derma.Application.SkinAnalysis;
using Derma.Domain;
using Derma.Infrastructure;
using MediatR;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using Derma.API.Models;

namespace Derma.API.Controllers
{
    public class SkinAnalysisController : BaseApiController
    {
        private readonly IMediator _mediator;
        public SkinAnalysisController(IMediator mediator)
        {
            _mediator = mediator;
        }

        [HttpGet]
        public async Task<ActionResult<List<SkinAnalysis>>> GetSkinAnalysis() {
            return await _mediator.Send(new SkinAnalysisList.Query());
        }
        [HttpGet("{id}")]
        public async Task<ActionResult<SkinAnalysis>> GetSkinAnalysis(Guid id)
        {
            return await Mediator.Send(new SkinAnalysisDetails.Query { Id = id });
        }
        [HttpPost]
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
                userId = Guid.Empty,   // TODO: JWT
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
        public async Task<IActionResult> EditSkinAnalysis(Guid id, SkinAnalysis skinAnalysis)
        {
            skinAnalysis.Id = id;
            await Mediator.Send(new SkinAnalysisEdit.Command { SkinAnalysis = skinAnalysis });
            return Ok();
        }
        [HttpDelete("{id}")]
        public async Task<IActionResult> DeleteSkinAnalysis(Guid id)
        {
            await Mediator.Send(
                new SkinAnalysisDelete.Command { Id = id });

            return Ok();
        }
    }
}
